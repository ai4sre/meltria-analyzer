import argparse
import json
import os
import random
import sys
import time
import warnings
from concurrent import futures
from datetime import datetime
from typing import Any, Callable

import banpei
import numpy as np
import pandas as pd
import scipy.stats
from arch.unitroot import PhillipsPerron
from arch.utility.exceptions import InfeasibleTestException
from lib.metrics import ROOT_METRIC_LABEL, check_cause_metrics
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.stattools import adfuller

from tsdr.clustering.kshape import kshape
from tsdr.clustering.metricsnamecluster import cluster_words
from tsdr.clustering.sbd import sbd, silhouette_score
from tsdr.outlierdetection.ar import AROutlierDetector
from tsdr.outlierdetection.knn import KNNOutlierDetector
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


class UnivariateSeriesReductionResult:
    _original_series: np.ndarray
    _has_kept: bool
    _anomaly_scores: np.ndarray
    _abn_th: float

    def __init__(
        self,
        original_series: np.ndarray,
        has_kept: bool,
        anomaly_scores: np.ndarray = np.array([]),
        abn_th: float = 0.0,
    ) -> None:
        self._original_series = original_series
        self._has_kept = has_kept
        self._anomaly_scores = anomaly_scores
        self._abn_th = abn_th

    @property
    def original_series(self):
        return self._original_series

    @property
    def has_kept(self):
        return self._has_kept

    @property
    def anomaly_scores(self):
        return self._anomaly_scores

    def binary_scores(self) -> np.ndarray:
        bin_scores = np.empty(self.anomaly_scores.size, dtype=np.uint8)
        bin_scores[np.argwhere(self.anomaly_scores <= self._abn_th)] = 0
        bin_scores[np.argwhere(self.anomaly_scores > self._abn_th)] = 1
        return bin_scores


def unit_root_based_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    regression: str = kwargs.get('tsifter_step1_unit_root_regression', 'c')
    maxlag: int = kwargs.get('tsifter_step1_unit_root_max_lags', None)
    autolag = kwargs.get('tsifter_step1_unit_root_autolag', None)

    if kwargs.get('tsifter_step1_pre_cv', False):
        cv_threshold = kwargs.get('tsifter_step1_cv_threshold', 0.01)
        if not has_variation(np.diff(series), cv_threshold) or not has_variation(series, cv_threshold):
            return UnivariateSeriesReductionResult(series, has_kept=True)

    def log_or_nothing(x: np.ndarray) -> np.ndarray:
        if kwargs.get('tsifter_step1_take_log', False):
            return np.log1p(x)
        return x

    pvalue: float = 0.0
    ar_lag: int = 0
    if kwargs['tsifter_step1_unit_root_model'] == 'adf':
        pvalue, ar_lag = adfuller(x=log_or_nothing(series), regression=regression, maxlag=maxlag, autolag=autolag)[1::2]
    elif kwargs['tsifter_step1_unit_root_model'] == 'pp':
        try:
            pp = PhillipsPerron(log_or_nothing(series), trend=regression, lags=maxlag)
            pvalue = pp.pvalue
            ar_lag = pp.lags
        except ValueError as e:
            warnings.warn(str(e))
            return UnivariateSeriesReductionResult(series, has_kept=False)
        except InfeasibleTestException as e:
            warnings.warn(str(e))
            return UnivariateSeriesReductionResult(series, has_kept=False)
    if pvalue >= kwargs.get('tsifter_step1_unit_root_alpha', 0.01):
        return UnivariateSeriesReductionResult(series, has_kept=True)
    else:
        # Post outlier detection
        odmodel: str = kwargs.get('tsifter_step1_post_od_model', 'knn')
        if odmodel == 'knn':
            knn = KNNOutlierDetector(w=ar_lag, k=1)   # k=1
            x = scipy.stats.zscore(log_or_nothing(series))
            if knn.has_anomaly(x, kwargs.get('tsifter_step1_post_od_threshold', 3.0)):
                return UnivariateSeriesReductionResult(series, has_kept=True)
        elif odmodel == 'hotelling':
            outliers = banpei.Hotelling().detect(series, kwargs.get('tsifter_step1_post_od_threshold', 0.01))
            if len(outliers) > 1:
                return UnivariateSeriesReductionResult(series, has_kept=True)
        else:
            raise ValueError(f"{odmodel} == 'knn' or 'hotelling'")
    return UnivariateSeriesReductionResult(series, has_kept=False)


def ar_based_ad_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    cv_threshold = kwargs.get('tsifter_step1_cv_threshold', 0.01)
    if not has_variation(np.diff(series), cv_threshold) or not has_variation(series, cv_threshold):
        return UnivariateSeriesReductionResult(series, has_kept=False)

    ar_threshold: float = kwargs.get('tsifter_step1_ar_anomaly_score_threshold', 0.01)
    ar = AROutlierDetector()
    scores: np.ndarray = ar.score(
        x=series,
        regression=kwargs.get('tsifter_step1_ar_regression', 'c'),
        dynamic_prediction=kwargs.get('tsifter_step1_ar_dynamic_prediction', False)
    )[0]
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"scores must contain only finite values. {scores}")
    outliers, abn_th = ar.detect_by_fitting_dist(scores, threshold=ar_threshold)
    if len(outliers) > 0:
        return UnivariateSeriesReductionResult(series, has_kept=True, anomaly_scores=scores, abn_th=abn_th)
    return UnivariateSeriesReductionResult(series, has_kept=False, anomaly_scores=scores, abn_th=abn_th)


class Tsdr:
    params: dict[str, Any]

    def __init__(
        self,
        univariate_series_func: Callable[[np.ndarray, Any], UnivariateSeriesReductionResult],
        **kwargs
    ) -> None:
        setattr(self, 'univariate_series_func', univariate_series_func)
        self.params = kwargs

    def univariate_series_func(self, series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
        return ar_based_ad_model(series, **kwargs)

    def run(
        self,
        series: pd.DataFrame,
        max_workers: int,
    ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
        services: list[str] = prepare_services_list(series)
        metrics_dimension: dict[str, Any] = aggregate_dimension(series)

        # step1
        start: float = time.time()

        reduced_series1, step1_results = self.reduce_univariate_series(series, max_workers)

        time_adf: float = round(time.time() - start, 2)
        metrics_dimension = util.count_metrics(
            metrics_dimension, reduced_series1, 1)
        metrics_dimension["total"].append(len(reduced_series1.columns))

        # step2
        df_before_clustering: pd.DataFrame = pd.DataFrame()
        series_type = self.params['tsifter_step2_clustered_series_type']
        if series_type == 'raw':
            df_before_clustering = reduced_series1
        elif series_type == 'anomaly_score':
            for name, res in step1_results.items():
                if res.has_kept:
                    df_before_clustering[name] = res.anomaly_scores
        elif series_type == 'binary_anomaly_score':
            for name, res in step1_results.items():
                if res.has_kept:
                    df_before_clustering[name] = res.binary_scores()
        else:
            raise ValueError(f'tsifter_step2_clustered_series_type is invalid {series_type}')

        start = time.time()

        reduced_series2, clustering_info = self.reduce_multivariate_series(
            df_before_clustering.copy(), services, max_workers,
            self.params['tsifter_step2_clustering_threshold'],
        )

        time_clustering: float = round(time.time() - start, 2)
        metrics_dimension = util.count_metrics(metrics_dimension, reduced_series2, 2)
        metrics_dimension["total"].append(len(reduced_series2.columns))

        return {'step1': time_adf, 'step2': time_clustering}, \
            {'step1': df_before_clustering, 'step2': reduced_series2}, metrics_dimension, clustering_info

    def reduce_univariate_series(
        self,
        useries: pd.DataFrame,
        n_workers: int,
    ) -> tuple[pd.DataFrame, dict[str, UnivariateSeriesReductionResult]]:
        results: dict[str, UnivariateSeriesReductionResult] = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_col = {}
            for col in useries.columns:
                series: np.ndarray = useries[col].to_numpy()
                if series.sum() == 0. or len(np.unique(series)) == 1 or np.isnan(series.sum()):
                    continue
                future = executor.submit(self.univariate_series_func, series, **self.params)
                future_to_col[future] = col
            reduced_cols: list[str] = []
            for future in futures.as_completed(future_to_col):
                col = future_to_col[future]
                result: UnivariateSeriesReductionResult = future.result()
                results[col] = result
                if result.has_kept:
                    reduced_cols.append(col)
        return useries[reduced_cols], results

    def reduce_multivariate_series(
        self,
        series: pd.DataFrame,
        services: list[str],
        n_workers: int,
        dist_threshold: float,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        clustering_info: dict[str, Any] = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Clustering metrics by service including services, containers and middlewares metrics
            future_list = []
            for ser in services:
                # perform clustering in each type of metric
                service_metrics_df = series.loc[:, series.columns.str.startswith(("s-{}_".format(ser)))]
                container_metrics_df = series.loc[:, series.columns.str.startswith(("c-{}_".format(ser)))]
                middleware_metrics_df = series.loc[:, series.columns.str.startswith(("m-{}_".format(ser), "m-{}-".format(ser)))]
                for target_df in [service_metrics_df, container_metrics_df, middleware_metrics_df]:
                    if len(target_df.columns) <= 1:
                        continue
                    future_list.append(executor.submit(
                        hierarchical_clustering, target_df, sbd, dist_threshold,
                    ))
            for future in futures.as_completed(future_list):
                c_info, remove_list = future.result()
                clustering_info.update(c_info)
                series = series.drop(remove_list, axis=1)
        return series, clustering_info


def has_variation(x: np.ndarray, cv_threshold) -> bool:
    mean = x.mean()
    std = x.std()
    if mean == 0.:
        # Differential series is possible to have zero mean.
        # see https://math.stackexchange.com/questions/1729033/calculating-the-variation-coefficient-when-the-arithmetic-mean-is-zero
        return False
    if mean == 0. and std == 0.:
        cv = 0
    else:
        cv = abs(std / mean)
    return cv > cv_threshold


def reduce_series_with_cv(data_df: pd.DataFrame, cv_threshold: float = 0.002):
    reduced_by_cv_df = pd.DataFrame()
    for col in data_df.columns:
        data = data_df[col].values
        if has_variation(data, cv_threshold):
            reduced_by_cv_df[col] = data_df[col]
    return reduced_by_cv_df


def hierarchical_clustering(
    target_df: pd.DataFrame, dist_func, dist_threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    series: np.ndarray = target_df.apply(scipy.stats.zscore).values.T
    dist = pdist(series, metric=dist_func)
    dist_matrix: np.ndarray = squareform(dist)
    z: np.ndarray = linkage(dist, method="single", metric=dist_func)
    labels: np.ndarray = fcluster(z, t=dist_threshold, criterion="distance")
    cluster_dict: dict[str, list[int]] = {}
    for i, v in enumerate(labels):
        if v in cluster_dict:
            cluster_dict[v].append(i)
        else:
            cluster_dict[v] = [i]

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
                    if met1 == met2:
                        continue
                    dist_sum += dist_matrix[met1][met2]
                distances.append(dist_sum)
            medoid = cluster_metrics[np.argmin(distances)]
            clustering_info[target_df.columns[medoid]] = []
            for r in cluster_metrics:
                if r == medoid:
                    continue
                remove_list.append(target_df.columns[r])
                clustering_info[target_df.columns[medoid]].append(
                    target_df.columns[r])
    return clustering_info, remove_list


def create_clusters(data: pd.DataFrame, columns: list[str], service_name: str, n: int):
    words_list: list[str] = [col[2:] for col in columns]
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


def select_representative_metric(
    data: pd.DataFrame,
    cluster_metrics: list[str], columns: dict[str, Any], centroid: int,
) -> tuple[dict[str, Any], list[str]]:
    clustering_info: dict[str, Any] = {}
    remove_list: list[str] = []
    if len(cluster_metrics) == 1:
        return clustering_info, remove_list
    if len(cluster_metrics) == 2:
        # Select the representative metric at random
        shuffle_list: list[str] = random.sample(cluster_metrics, len(cluster_metrics))
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
            if r == representative_metric:
                continue
            remove_list.append(columns[r])
            clustering_info[columns[representative_metric]].append(
                columns[r])
    return clustering_info, remove_list


def kshape_clustering(
    target_df: pd.DataFrame, service_name: str, executor,
) -> tuple[dict[str, Any], list[str]]:
    future_list = []

    data: np.ndarray = target_df.apply(scipy.stats.zscore).values.T
    for n in np.arange(2, data.shape[0]):
        future_list.append(
            executor.submit(
                create_clusters, data, target_df.columns, service_name, n,
            )
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


def sieve_reduce_series(data_df):
    return reduce_series_with_cv(data_df)


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


def run_sieve(
    data_df: pd.DataFrame,
    metrics_dimension: dict[str, Any],
    services_list: list[str],
    max_workers: int,
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


def prepare_services_list(data_df: pd.DataFrame) -> list[str]:
    # Prepare list of services
    services_list: list[str] = []
    for col in data_df.columns:
        if not col.startswith('s-'):
            continue
        service_name = col.split("_")[0].replace("s-", "")
        if service_name not in services_list:
            services_list.append(service_name)
    return services_list


def aggregate_dimension(data_df: pd.DataFrame) -> dict[str, Any]:
    metrics_dimension: dict[str, Any] = {}
    for target in TARGET_DATA:
        metrics_dimension[target] = {}
    metrics_dimension = util.count_metrics(metrics_dimension, data_df, 0)
    metrics_dimension["total"] = [len(data_df.columns)]
    return metrics_dimension


def run_tsdr(data_df: pd.DataFrame, method: str, max_workers: int, **kwargs,
             ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    if method == TSIFTER_METHOD:
        tsdr = Tsdr(**kwargs)
        return tsdr.run(data_df, max_workers)
    elif method == SIEVE_METHOD:
        services: list[str] = prepare_services_list(data_df)
        metrics_dimension: dict[str, Any] = aggregate_dimension(data_df)
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
    parser.add_argument("--tsifter-cv-threshold",
                        type=float,
                        default=0.05,
                        help='CV threshold for tsifter')
    parser.add_argument("--tsifter-clustering-threshold",
                        type=float,
                        default=THRESHOLD_DIST,
                        help='distance threshold for hierarchical clustering')
    args = parser.parse_args()

    data_df, mappings, metrics_meta = read_metrics_json(args.datafile)
    elapsedTime, reduced_df_by_step, metrics_dimension, clustering_info = run_tsdr(
        data_df=data_df,
        method=args.method,
        max_workers=args.max_workers,
        tsifter_step1_unit_root_alpha=args.tsifter_adf_alpha,
        tsifter_step1_cv_threshold=args.tsifter_cv_threshold,
        tsifter_step1_knn_threshold=args.tsifter_knn_threshold,
        tsifter_step2_clustering_threshold=args.tsifter_clustering_threshold,
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
