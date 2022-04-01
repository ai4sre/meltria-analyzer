import json
import random
import time
import warnings
from concurrent import futures
from typing import Any, Callable

import banpei
import numpy as np
import pandas as pd
import scipy.ndimage as ndimg
import scipy.stats
from arch.unitroot import PhillipsPerron
from arch.utility.exceptions import InfeasibleTestException
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import hamming, pdist, squareform
from statsmodels.tsa.stattools import adfuller
from tsmoothie.smoother import BinnerSmoother

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
    def __init__(
        self,
        original_series: np.ndarray,
        has_kept: bool,
        anomaly_scores: np.ndarray = np.array([]),
        abn_th: float = 0.0,
        outliers: list[tuple[int, float]] = [],
    ) -> None:
        self._original_series: np.ndarray = original_series
        self._has_kept: bool = has_kept
        self._anomaly_scores: np.ndarray = anomaly_scores
        self._abn_th: float = abn_th
        self._outliers: np.ndarray = np.array(outliers, dtype=object)  # mix int and float

    @property
    def original_series(self) -> np.ndarray:
        return self._original_series

    @property
    def has_kept(self) -> bool:
        return self._has_kept

    @property
    def anomaly_scores(self) -> np.ndarray:
        return self._anomaly_scores

    @property
    def outliers(self) -> np.ndarray:
        return self._outliers

    def binary_scores(self) -> np.ndarray:
        bin_scores = np.empty(self.anomaly_scores.size, dtype=np.uint8)
        bin_scores[np.argwhere(self.anomaly_scores <= self._abn_th)] = 0
        bin_scores[np.argwhere(self.anomaly_scores > self._abn_th)] = 1
        return bin_scores


def detect_with_cv(series: np.ndarray, **kwargs: Any) -> bool:
    cv_threshold = kwargs.get('tsifter_step1_cv_threshold', 0.01)
    return not has_variation(np.diff(series), cv_threshold) or not has_variation(series, cv_threshold)


def cv_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    return UnivariateSeriesReductionResult(series, has_kept=(not detect_with_cv(series, **kwargs)))


def unit_root_based_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    regression: str = kwargs.get('tsifter_step1_unit_root_regression', 'c')
    maxlag: int = kwargs.get('tsifter_step1_unit_root_max_lags', None)
    autolag = kwargs.get('tsifter_step1_unit_root_autolag', None)

    if kwargs.get('tsifter_step1_pre_cv', False):
        if detect_with_cv(series, **kwargs):
            return UnivariateSeriesReductionResult(series, has_kept=False)

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


def ar_based_ad_model(orig_series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if detect_with_cv(orig_series, **kwargs):
        return UnivariateSeriesReductionResult(orig_series, has_kept=False)

    if (smoother := kwargs.get('tsifter_step1_smoother')) is not None:
        if smoother == 'none':
            series = orig_series
        elif smoother == 'binner':
            series = smooth_with_binner(orig_series, **kwargs)
        elif smoother == 'moving_average':
            series = smooth_with_ma(orig_series, **kwargs)
        else:
            raise ValueError(f"Invalid smoother: '{smoother}'")
    else:
        series = orig_series

    ar_threshold: float = kwargs.get('tsifter_step1_ar_anomaly_score_threshold', 0.01)
    ar_lag: int = kwargs.get('tsifter_step1_ar_lag', 0)
    ar = AROutlierDetector(maxlag=ar_lag)
    scores: np.ndarray = ar.score(
        x=series,
        regression=kwargs.get('tsifter_step1_ar_regression', 'n'),
        lag=ar_lag,
        autolag=True if ar_lag == 0 else False,
        ic=kwargs.get('tsifter_step1_ar_ic', 'bic'),
        dynamic_prediction=kwargs.get('tsifter_step1_ar_dynamic_prediction', False),
    )[0]
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"scores must contain only finite values. {scores}")
    outliers, abn_th = AROutlierDetector.detect_by_fitting_dist(scores, threshold=ar_threshold)
    if len(outliers) > 0:
        return UnivariateSeriesReductionResult(
            orig_series, has_kept=True, anomaly_scores=scores, abn_th=abn_th, outliers=outliers)
    return UnivariateSeriesReductionResult(orig_series, has_kept=False, anomaly_scores=scores, abn_th=abn_th)


def hotteling_t2_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if detect_with_cv(series):
        return UnivariateSeriesReductionResult(series, has_kept=False)

    outliers = banpei.Hotelling().detect(series, kwargs.get('tsifter_step1_hotteling_threshold', 0.01))
    if len(outliers) > 1:
        return UnivariateSeriesReductionResult(series, has_kept=True, outliers=outliers)
    return UnivariateSeriesReductionResult(series, has_kept=False)


def sst_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if detect_with_cv(series, **kwargs):
        return UnivariateSeriesReductionResult(series, has_kept=False)

    sst = banpei.SST(w=len(series)//2)
    change_scores: np.ndarray = sst.detect(scipy.stats.zscore(series), is_lanczos=True)
    change_pts: list[tuple[int, float]] = []
    for i, score in enumerate(change_scores):
        if score >= kwargs.get('tsifter_step1_sst_threshold'):
            change_pts.append((i, score))
    if len(change_pts) > 0:
        return UnivariateSeriesReductionResult(series, has_kept=True, anomaly_scores=change_scores, outliers=change_pts)
    return UnivariateSeriesReductionResult(series, has_kept=False, anomaly_scores=change_scores)


def smooth_with_ma(x: np.ndarray, **kwargs: Any) -> np.ndarray:
    w: int = kwargs.get('tsifter_step1_ma_window_size', 2)
    return ndimg.uniform_filter1d(input=x, size=w, mode='constant', origin=-(w//2))[:-(w-1)]


def smooth_with_binner(x: np.ndarray, **kwargs: Any) -> np.ndarray:
    """ Smooth time series with binner method.
    """
    w: int = kwargs.get('tsifter_step1_smoother_binner_window_size', 2)
    smoother = BinnerSmoother(n_knots=int(x.size/w), copy=True)
    smoother.smooth(x)
    return smoother.smooth_data[0]


class Tsdr:
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
    ) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
        metrics_dimension: dict[str, Any] = aggregate_dimension(series)

        # step1
        start: float = time.time()

        reduced_series1, step1_results, anomaly_points = self.reduce_univariate_series(series, max_workers)

        time_adf: float = round(time.time() - start, 2)
        metrics_dimension = util.count_metrics(
            metrics_dimension, reduced_series1, 1)
        metrics_dimension["total"].append(len(reduced_series1.columns))

        # step2
        df_before_clustering: pd.DataFrame
        series_type = self.params['tsifter_step2_clustered_series_type']
        if series_type == 'raw':
            df_before_clustering = reduced_series1.apply(scipy.stats.zscore)
        elif series_type in ['anomaly_score', 'binary_anomaly_score']:
            tmp_dict_to_df: dict[str, np.ndarray] = {}
            for name, res in step1_results.items():
                if res.has_kept:
                    if series_type == 'anomaly_score':
                        tmp_dict_to_df[name] = scipy.stats.zscore(res.anomaly_scores)
                    elif series_type == 'binary_anomaly_score':
                        tmp_dict_to_df[name] = res.binary_scores()
            df_before_clustering = pd.DataFrame(tmp_dict_to_df)
        else:
            raise ValueError(f'tsifter_step2_clustered_series_type is invalid {series_type}')

        containers_of_service: dict[str, set[str]] = get_container_names_of_service(series)

        start = time.time()

        reduced_series2, clustering_info = self.reduce_multivariate_series(
            df_before_clustering.copy(), containers_of_service, max_workers,
            self.params['tsifter_step2_clustering_dist_type'],
            self.params['tsifter_step2_clustering_threshold'],
            self.params['tsifter_step2_clustering_choice_method'],
            self.params['tsifter_step2_clustering_linkage_method'],
        )

        time_clustering: float = round(time.time() - start, 2)
        metrics_dimension = util.count_metrics(metrics_dimension, reduced_series2, 2)
        metrics_dimension["total"].append(len(reduced_series2.columns))

        return {'step1': time_adf, 'step2': time_clustering}, \
            {'step1': df_before_clustering, 'step2': reduced_series2}, \
            metrics_dimension, clustering_info, anomaly_points

    def reduce_univariate_series(
        self,
        useries: pd.DataFrame,
        n_workers: int,
    ) -> tuple[pd.DataFrame, dict[str, UnivariateSeriesReductionResult], dict[str, np.ndarray]]:
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
        anomaly_points = {col: res.outliers for col, res in results.items()}
        return useries[reduced_cols], results, anomaly_points

    def reduce_multivariate_series(
        self,
        series: pd.DataFrame,
        containers_of_service: dict[str, set[str]],
        n_workers: int,
        dist_type: str,
        dist_threshold: float,
        choice_method: str,
        linkage_method: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        def make_clusters(
            df: pd.DataFrame,
            dist_threshold: float,
            choice_method: str,
            linkage_method: str,
        ) -> futures.Future:
            future: futures.Future
            if dist_type == 'sbd':
                future = executor.submit(
                    hierarchical_clustering,
                    df, sbd, dist_threshold, choice_method, linkage_method,
                )
            elif dist_type == 'hamming':
                if dist_threshold >= 1.0:
                    # make the distance threshold intuitive
                    dist_threshold /= series.shape[0]
                future = executor.submit(
                    hierarchical_clustering,
                    df, hamming, dist_threshold, choice_method, linkage_method,
                )
            else:
                raise ValueError('dist_func must be "sbd" or "hamming"')
            return future

        clustering_info: dict[str, Any] = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Clustering metrics by service including services, containers and middlewares metrics
            future_list: list[futures.Future] = []
            for service, containers in containers_of_service.items():
                service_metrics_df = series.loc[:, series.columns.str.startswith(f"s-{service}_")]
                if len(service_metrics_df.columns) > 1:
                    future_list.append(
                        make_clusters(service_metrics_df, dist_threshold, choice_method, linkage_method),
                    )
                for container in containers:
                    # perform clustering in each type of metric
                    container_metrics_df = series.loc[:, series.columns.str.startswith(f"c-{container}_")]
                    # TODO: middleware
                    # middleware_metrics_df = series.loc[:, series.columns.str.startswith(("m-{}_".format(ser), "m-{}-".format(ser)))]
                    if len(container_metrics_df.columns) <= 1:
                        continue
                    future_list.append(
                        make_clusters(container_metrics_df, dist_threshold, choice_method, linkage_method),
                    )
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
    target_df: pd.DataFrame,
    dist_func: Callable,
    dist_threshold: float,
    choice_method: str = 'medoid',
    linkage_method: str = 'single',
) -> tuple[dict[str, Any], list[str]]:
    dist = pdist(target_df.values.T, metric=dist_func)
    dist_matrix: np.ndarray = squareform(dist)
    z: np.ndarray = linkage(dist, method=linkage_method, metric=dist_func)
    labels: np.ndarray = fcluster(z, t=dist_threshold, criterion="distance")
    cluster_dict: dict[str, list[int]] = {}
    for i, v in enumerate(labels):
        if v in cluster_dict:
            cluster_dict[v].append(i)
        else:
            cluster_dict[v] = [i]

    if choice_method == 'medoid':
        return choose_metric_with_medoid(target_df.columns, cluster_dict, dist_matrix)
    elif choice_method == 'maxsum':
        return choose_metric_with_maxsum(target_df, cluster_dict)
    else:
        raise ValueError('choice_method is required.')


def choose_metric_with_medoid(
    columns: pd.Index,
    cluster_dict: dict[str, list[int]],
    dist_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics = cluster_dict[c]
        if len(cluster_metrics) == 1:
            continue
        if len(cluster_metrics) == 2:
            # Select the representative metric at random
            shuffle_list = random.sample(cluster_metrics, len(cluster_metrics))
            clustering_info[columns[shuffle_list[0]]] = [
                columns[shuffle_list[1]]]
            remove_list.append(columns[shuffle_list[1]])
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
            clustering_info[columns[medoid]] = []
            for r in cluster_metrics:
                if r == medoid:
                    continue
                remove_list.append(columns[r])
                clustering_info[columns[medoid]].append(columns[r])
    return clustering_info, remove_list


def choose_metric_with_maxsum(
    data_df: pd.DataFrame,
    cluster_dict: dict[str, list[int]],
) -> tuple[dict[str, Any], list[str]]:
    """ Choose metrics which has max of sum of datapoints in each metrics in each cluster. """
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics: list[int] = cluster_dict[c]
        if len(cluster_metrics) == 1:
            continue
        if len(cluster_metrics) > 1:
            cluster_columns = data_df.columns[cluster_metrics]
            series_with_sum: pd.Series = data_df[cluster_columns].sum(numeric_only=True)
            label_with_max: str = series_with_sum.idxmax()
            sub_metrics: list[str] = list(series_with_sum.loc[series_with_sum.index != label_with_max].index)
            clustering_info[label_with_max] = sub_metrics
            remove_list += sub_metrics
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


def get_container_names_of_service(data_df: pd.DataFrame) -> dict[str, set[str]]:
    """ get component (services and containers) names

    Returns:
        dict[str]: expected to be like libs.SERVICE_CONTAINERS
    """
    service_cols, container_cols = [], []
    for col in data_df.columns:
        if col.startswith('s-'):
            service_cols.append(col)
        if col.startswith('c-'):
            container_cols.append(col)
    # TODO: middleware

    components: dict[str, set[str]] = {}
    services: set[str] = set([])
    for service_col in service_cols:
        service_name = service_col.split('_')[0].replace('s-', '')
        components[service_name] = set([])
        services.add(service_name)
    for container_col in container_cols:
        container_name = container_col.split('_')[0].replace('c-', '')
        service_name = [s for s in services if container_name.startswith(s)][0]
        # container should be unique
        components[service_name].add(container_name)
    return components


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
