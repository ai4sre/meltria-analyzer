import argparse
import json
import os
import random
import re
import sys
import time
from concurrent import futures
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from lib.metrics import ROOT_METRIC_LABEL, check_cause_metrics
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.stattools import adfuller

from tsdr.clustering.kshape import kshape
from tsdr.clustering.metricsnamecluster import cluster_words
from tsdr.clustering.sbd import sbd, silhouette_score
from tsdr.util import util

TSIFTER_METHOD = 'tsifter'
SIEVE_METHOD = 'sieve'

PLOTS_NUM = 120
SIGNIFICANCE_LEVEL = 0.05
THRESHOLD_DIST = 0.01
TARGET_DATA = {"containers": "all",
               "services": "all",
               "nodes": "all",
               "middlewares": "all"}


def reduce_series_with_cv(data_df, cv_threshold=0.002):
    reduced_by_cv_df = pd.DataFrame()
    for col in data_df.columns:
        data = data_df[col].values
        mean = data.mean()
        std = data.std()
        if mean == 0. and std == 0.:
            cv = 0
        else:
            cv = std / mean
        if cv > cv_threshold:
            reduced_by_cv_df[col] = data_df[col]
    return reduced_by_cv_df


def hierarchical_clustering(target_df, dist_func, dist_threshold: float):
    series = target_df.values.T
    norm_series = util.z_normalization(series)
    dist = pdist(norm_series, metric=dist_func)
    # distance_list.extend(dist)
    dist_matrix = squareform(dist)
    z = linkage(dist, method="single", metric=dist_func)
    labels = fcluster(z, t=dist_threshold, criterion="distance")
    cluster_dict = {}
    for i, v in enumerate(labels):
        if v not in cluster_dict:
            cluster_dict[v] = [i]
        else:
            cluster_dict[v].append(i)
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics = cluster_dict[c]
        if len(cluster_metrics) == 1:
            continue
        if len(cluster_metrics) == 2:
            # Select the representative metric at random
            shuffle_list = random.sample(cluster_metrics, len(cluster_metrics))
            clustering_info[target_df.columns[shuffle_list[0]]] = [
                target_df.columns[shuffle_list[1]]]
            remove_list.append(target_df.columns[shuffle_list[1]])
        elif len(cluster_metrics) > 2:
            # Select medoid as the representative metric
            distances = []
            for met1 in cluster_metrics:
                dist_sum = 0
                for met2 in cluster_metrics:
                    if met1 != met2:
                        dist_sum += dist_matrix[met1][met2]
                distances.append(dist_sum)
            medoid = cluster_metrics[np.argmin(distances)]
            clustering_info[target_df.columns[medoid]] = []
            for r in cluster_metrics:
                if r != medoid:
                    remove_list.append(target_df.columns[r])
                    clustering_info[target_df.columns[medoid]].append(
                        target_df.columns[r])
    return clustering_info, remove_list


def create_clusters(data, columns, service_name, n):
    words_list = [col[2:] for col in columns]
    init_labels = cluster_words(words_list, service_name, n)
    results = kshape(data, n, initial_clustering=init_labels)
    label = [0] * data.shape[0]
    cluster_center = []
    cluster_num = 0
    for res in results:
        if not res[1]:
            continue
        for i in res[1]:
            label[i] = cluster_num
        cluster_center.append(res[0])
        cluster_num += 1
    if len(set(label)) == 1:
        return None
    return (label, silhouette_score(data, label), cluster_center)


def select_representative_metric(data, cluster_metrics, columns, centroid):
    clustering_info = {}
    remove_list = []
    if len(cluster_metrics) == 1:
        return None, None
    if len(cluster_metrics) == 2:
        # Select the representative metric at random
        shuffle_list = random.sample(cluster_metrics, len(cluster_metrics))
        clustering_info[columns[shuffle_list[0]]] = [columns[shuffle_list[1]]]
        remove_list.append(columns[shuffle_list[1]])
    elif len(cluster_metrics) > 2:
        # Select the representative metric based on the distance from the centroid
        distances = []
        for met in cluster_metrics:
            distances.append(sbd(centroid, data[met]))
        representative_metric = cluster_metrics[np.argmin(distances)]
        clustering_info[columns[representative_metric]] = []
        for r in cluster_metrics:
            if r != representative_metric:
                remove_list.append(columns[r])
                clustering_info[columns[representative_metric]].append(
                    columns[r])
    return (clustering_info, remove_list)


def kshape_clustering(target_df, service_name, executor):
    future_list = []

    data = util.z_normalization(target_df.values.T)
    for n in np.arange(2, data.shape[0]):
        future_list.append(
            executor.submit(create_clusters, data,
                            target_df.columns, service_name, n)
        )
    labels, scores, centroids = [], [], []
    for future in futures.as_completed(future_list):
        cluster = future.result()
        if cluster is None:
            continue
        labels.append(cluster[0])
        scores.append(cluster[1])
        centroids.append(cluster[2])

    idx = np.argmax(scores)
    label = labels[idx]
    centroid = centroids[idx]
    cluster_dict = {}
    for i, v in enumerate(label):
        if v not in cluster_dict:
            cluster_dict[v] = [i]
        else:
            cluster_dict[v].append(i)

    future_list = []
    for c, cluster_metrics in cluster_dict.items():
        future_list.append(
            executor.submit(select_representative_metric, data,
                            cluster_metrics, target_df.columns, centroid[c])
        )
    clustering_info = {}
    remove_list = []
    for future in futures.as_completed(future_list):
        c_info, r_list = future.result()
        if c_info is None:
            continue
        clustering_info.update(c_info)
        remove_list.extend(r_list)

    return clustering_info, remove_list


def is_unstational_series(series: np.ndarray,
                          alpha: float,
                          regression: str = 'c',
                          maxlag: int = None,
                          autolag: str = None) -> Optional[float]:
    pvalue: float = adfuller(x=series, regression=regression, maxlag=maxlag, autolag=autolag)[1]
    if not np.isnan(pvalue) and pvalue >= alpha:
        return True
    return False

def tsifter_reduce_series(data_df: pd.DataFrame, max_workers: int,
                          step1_method: str, step1_alpha: float, step1_regression: str) -> pd.DataFrame:
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_col = {}
        for col in data_df.columns:
            series: np.ndarray = data_df[col].to_numpy()
            if series.sum() == 0. or len(np.unique(series)) == 1 or np.isnan(series.sum()):
                continue
            # run df-test for differences of data_{n} and data{n-1} for liner trend series
            if step1_method == 'adf':
                future = executor.submit(
                    is_unstational_series, series, step1_alpha, regression=step1_regression,
                )
                future_to_col[future] = col
            elif step1_method == 'df':
                future = executor.submit(
                    is_unstational_series,
                    series, step1_alpha, regression=step1_regression, maxlag=1, autolag=None,
                )
                future_to_col[future] = col
            else:
                raise ValueError('step1_method must be adf or df')
        reduced_cols: list[str] = []
        for is_unstationality in futures.as_completed(future_to_col):
            col = future_to_col[is_unstationality]
            if is_unstationality.result():
                reduced_cols.append(col)
    return data_df[reduced_cols]


def sieve_reduce_series(data_df):
    return reduce_series_with_cv(data_df)


def tsifter_clustering(reduced_df, services_list, max_workers, dist_threshold: float):
    clustering_info = {}
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Clustering metrics by service including services, containers and middlewares metrics
        future_list = []
        for ser in services_list:
            target_df = reduced_df.loc[:, reduced_df.columns.str.startswith(
                ("s-{}_".format(ser), "c-{}_".format(ser), "c-{}-".format(ser), "m-{}_".format(ser), "m-{}-".format(ser)))]
            if len(target_df.columns) in [0, 1]:
                continue
            future_list.append(executor.submit(
                hierarchical_clustering, target_df, sbd, dist_threshold))
        for future in futures.as_completed(future_list):
            c_info, remove_list = future.result()
            clustering_info.update(c_info)
            reduced_df = reduced_df.drop(remove_list, axis=1)

    return reduced_df, clustering_info


def sieve_clustering(reduced_df, services_list, max_workers):
    clustering_info = {}

    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Clustering metrics by services including services, containers and middlewares
        for ser in services_list:
            target_df = reduced_df.loc[:, reduced_df.columns.str.startswith(
                ("s-{}_".format(ser), "c-{}_".format(ser), "c-{}-".format(ser), "m-{}_".format(ser), "m-{}-".format(ser)))]
            if len(target_df.columns) in [0, 1]:
                continue
            c_info, remove_list = kshape_clustering(target_df, ser, executor)
            clustering_info.update(c_info)
            reduced_df = reduced_df.drop(remove_list, axis=1)

    return reduced_df, clustering_info


def run_tsifter(data_df, metrics_dimension, services_list, max_workers,
                step1_method: str, step1_alpha: float, step1_regression: str, dist_threshold: float
                ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    # step1
    start = time.time()

    reduced_by_st_df = tsifter_reduce_series(data_df, max_workers, step1_method, step1_alpha, step1_regression)

    time_adf = round(time.time() - start, 2)
    metrics_dimension = util.count_metrics(
        metrics_dimension, reduced_by_st_df, 1)
    metrics_dimension["total"].append(len(reduced_by_st_df.columns))

    # step2
    start = time.time()

    reduced_df, clustering_info = tsifter_clustering(
        reduced_by_st_df.copy(), services_list, max_workers, dist_threshold)

    time_clustering = round(time.time() - start, 2)
    metrics_dimension = util.count_metrics(metrics_dimension, reduced_df, 2)
    metrics_dimension["total"].append(len(reduced_df.columns))

    return {'step1': time_adf, 'step2': time_clustering}, \
        {'step1': reduced_by_st_df, 'step2': reduced_df}, metrics_dimension, clustering_info


def run_sieve(data_df, metrics_dimension, services_list, max_workers
              ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    # step1
    start = time.time()

    reduced_by_st_df = sieve_reduce_series(data_df)

    time_cv = round(time.time() - start, 2)
    metrics_dimension = util.count_metrics(
        metrics_dimension, reduced_by_st_df, 1)
    metrics_dimension["total"].append(len(reduced_by_st_df.columns))

    # step2
    start = time.time()

    reduced_df, clustering_info = sieve_clustering(
        reduced_by_st_df.copy(), services_list, max_workers)

    time_clustering = round(time.time() - start, 2)
    metrics_dimension = util.count_metrics(metrics_dimension, reduced_df, 2)
    metrics_dimension["total"].append(len(reduced_df.columns))

    return {'step1': time_cv, 'step2': time_clustering}, \
        {'step1': reduced_by_st_df, 'step2': reduced_df}, metrics_dimension, clustering_info


def read_metrics_json(data_file: str,
                      interporate: bool = True,
                      exclude_middlewares: bool = False,
                      ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """ Read metrics data file """

    with open(data_file) as f:
        raw_json = json.load(f)
    raw_data = pd.read_json(data_file)
    data_df = pd.DataFrame()
    metrics_name_to_values: dict[str, np.ndarray] = {}
    for target in TARGET_DATA:
        for t in raw_data[target].dropna():
            for metric in t:
                if metric["metric_name"] not in TARGET_DATA[target] and TARGET_DATA[target] != "all":
                    continue
                if target == 'middlewares' and exclude_middlewares:
                    continue
                metric_name = metric["metric_name"].replace(
                    "container_", "").replace("node_", "")
                target_name = metric[
                    "{}_name".format(
                        target[:-1]) if target != "middlewares" else "container_name"
                ]
                if target_name in ["queue-master", "rabbitmq", "session-db"]:
                    continue
                metric_name = "{}-{}_{}".format(target[0],
                                                target_name, metric_name)
                metrics_name_to_values[metric_name] = np.array(
                    metric["values"], dtype=np.float64,
                )[:, 1][-PLOTS_NUM:]
    data_df = pd.DataFrame(metrics_name_to_values).round(4)
    if interporate:
        try:
            data_df = data_df.interpolate(
                method="spline", order=3, limit_direction="both")
        except:  # To cacth `dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=3`
            raise ValueError("calculating spline error") from None
    return data_df, raw_json['mappings'], raw_json['meta']


def prepare_services_list(data_df):
    # Prepare list of services
    services_list = []
    for col in data_df.columns:
        if re.match("^s-", col):
            service_name = col.split("_")[0].replace("s-", "")
            if service_name not in services_list:
                services_list.append(service_name)
    return services_list


def aggregate_dimension(data_df):
    metrics_dimension = {}
    for target in TARGET_DATA:
        metrics_dimension[target] = {}
    metrics_dimension = util.count_metrics(metrics_dimension, data_df, 0)
    metrics_dimension["total"] = [len(data_df.columns)]
    return metrics_dimension


def run_tsdr(data_df: pd.DataFrame, method: str, max_workers: int, **kwargs
             ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    services = prepare_services_list(data_df)
    metrics_dimension = aggregate_dimension(data_df)
    if method == TSIFTER_METHOD:
        return run_tsifter(
            data_df, metrics_dimension, services, max_workers,
            kwargs['tsifter_step1_method'],
            kwargs['tsifter_step1_alpha'],
            kwargs['tsifter_step1_regression'],
            kwargs['tsifter_clustering_threshold'],
        )
    elif method == SIEVE_METHOD:
        return run_sieve(data_df, metrics_dimension, services, max_workers)
    return {}, {}, {}, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="metrics JSON data file")
    parser.add_argument("--method",
                        choices=[TSIFTER_METHOD, SIEVE_METHOD],
                        help="specify one of tsdr methods",
                        default=TSIFTER_METHOD)
    parser.add_argument("--max-workers",
                        help="number of processes",
                        type=int, default=1)
    parser.add_argument("--plot-num",
                        help="number of plots",
                        type=int, default=PLOTS_NUM)
    parser.add_argument("--metric-num",
                        help="number of metrics (for experiment)",
                        type=int, default=None)
    parser.add_argument("--out", help="output path", type=str)
    parser.add_argument("--results-dir",
                        help="output directory",
                        action='store_true')
    parser.add_argument("--include-raw-data",
                        help="include time series to results",
                        action='store_true')
    parser.add_argument("--tsifter-adf-alpha",
                        type=float,
                        default=SIGNIFICANCE_LEVEL,
                        help='sigificance level for ADF test')
    parser.add_argument("--tsifter-clustering-threshold",
                        type=float,
                        default=THRESHOLD_DIST,
                        help='distance threshold for hierachical clustering')
    args = parser.parse_args()

    data_df, mappings, metrics_meta = read_metrics_json(args.datafile)
    elapsedTime, reduced_df_by_step, metrics_dimension, clustering_info = run_tsdr(
        data_df=data_df,
        method=args.method,
        max_workers=args.max_workers,
        tsifter_step1_method='df',
        tsifter_step1_alpha=args.tsifter_adf_alpha,
        tsifter_clustering_threshold=args.tsifter_clustering_threshold,
    )

    reduced_df = reduced_df_by_step['step2']  # final result

    # Check that the results include SLO metric
    root_metrics: list[str] = []
    for column in list(reduced_df.columns):
        if column == ROOT_METRIC_LABEL:
            root_metrics.append(column)

    # Check that the results include cause metric
    _, cause_metrics = check_cause_metrics(
        list(reduced_df.columns),
        metrics_meta['injected_chaos_type'],
        metrics_meta['chaos_injected_component'],
    )

    summary = {
        'tsdr_method': args.method,
        'data_file': args.datafile.split("/")[-1],
        'number_of_plots': PLOTS_NUM,
        'label_checking_results': {
            'root_metrics': root_metrics,
            'cause_metrics': cause_metrics,
        },
        'execution_time': {
            "reduce_series": elapsedTime['step1'],
            "clustering": elapsedTime['step2'],
            "total": round(elapsedTime['step1'] + elapsedTime['step2'], 2)
        },
        'metrics_dimension': metrics_dimension,
        'reduced_metrics': list(reduced_df.columns),
        'clustering_info': clustering_info,
        'components_mappings': mappings,
        'metrics_meta': metrics_meta,
    }
    if args.include_raw_data:
        summary["reduced_metrics_raw_data"] = reduced_df.to_dict()

    if args.results_dir:
        file_name = "{}_{}.json".format(
            TSIFTER_METHOD, datetime.now().strftime("%Y%m%d%H%M%S"))
        result_dir = "./results/{}".format(args.datafile.split("/")[-1])
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, file_name), "w") as f:
            json.dump(summary, f, indent=4)

    # print out, too.
    if args.out is None:
        json.dump(summary, sys.stdout)
    else:
        with open(args.out, mode='w') as f:
            json.dump(summary, f)
