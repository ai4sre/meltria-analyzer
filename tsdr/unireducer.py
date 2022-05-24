import warnings
from typing import Any, Callable

import banpei
import numpy as np
import ruptures as rpt
import scipy.ndimage as ndimg
import scipy.signal
import scipy.stats
from arch.unitroot import PhillipsPerron
from arch.utility.exceptions import InfeasibleTestException
from statsmodels.tsa.stattools import adfuller, kpss
from tsmoothie.smoother import BinnerSmoother

from tsdr.outlierdetection.ar import AROutlierDetector
from tsdr.outlierdetection.fluxinfer import FluxInferAD
from tsdr.outlierdetection.knn import KNNOutlierDetector
from tsdr.outlierdetection.residual_integral import residual_integral_max


class UnivariateSeriesReductionResult:
    def __init__(
        self,
        original_series: np.ndarray,
        has_kept: bool,
        anomaly_scores: np.ndarray = np.array([]),
        abn_th: float = 0.0,
        outliers: list[tuple[int, float]] = [],
        change_start_point: tuple[int, float] = (0, 0.0),
    ) -> None:
        self._original_series: np.ndarray = original_series
        self._has_kept: bool = has_kept
        self._anomaly_scores: np.ndarray = anomaly_scores
        self._abn_th: float = abn_th
        self._outliers: np.ndarray = np.array(outliers, dtype=object)  # mix int and float
        self._change_start_point = change_start_point

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

    @property
    def change_start_point(self) -> tuple[int, float]:
        return self._change_start_point

    def binary_scores(self) -> np.ndarray:
        bin_scores = np.empty(self.anomaly_scores.size, dtype=np.uint8)
        bin_scores[np.argwhere(self.anomaly_scores <= self._abn_th)] = 0
        bin_scores[np.argwhere(self.anomaly_scores > self._abn_th)] = 1
        return bin_scores


def map_model_name_to_func(model_name: str) -> Callable:
    match model_name:
        case 'cv':
            return cv_model
        case 'unit_root_test':
            return unit_root_based_model
        case 'ar_based_ad':
            return ar_based_ad_model
        case 'hotteling_t2':
            return hotteling_t2_model
        case 'sst':
            return sst_model
        case 'differencial_of_anomaly_score':
            return differencial_of_anomaly_score_model
        case 'fluxinfer':
            return fluxinfer_model
        case 'hist_and_stationality':
            return hist_and_stationality_model
        case 'residual_integral':
            return residual_integral_model
        case 'two_samp_test':
            return two_samp_test_model
        case _:
            raise ValueError(f'Invalid univariate time series model: {model_name}')


def detect_with_cv(series: np.ndarray, **kwargs: Any) -> bool:
    cv_threshold = kwargs.get('step1_cv_threshold', 0.01)
    return not has_variation(np.diff(series), cv_threshold) or not has_variation(series, cv_threshold)


def cv_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    return UnivariateSeriesReductionResult(series, has_kept=(not detect_with_cv(series, **kwargs)))


def unit_root_based_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    regression: str = kwargs.get('step1_unit_root_regression', 'c')
    maxlag: int = kwargs.get('step1_unit_root_max_lags', None)
    autolag = kwargs.get('step1_unit_root_autolag', None)

    if kwargs.get('step1_pre_cv', False):
        if detect_with_cv(series, **kwargs):
            return UnivariateSeriesReductionResult(series, has_kept=False)

    def log_or_nothing(x: np.ndarray) -> np.ndarray:
        if kwargs.get('step1_take_log', False):
            return np.log1p(x)
        return x

    pvalue: float = 0.0
    ar_lag: int = 0
    if kwargs['step1_unit_root_model'] == 'adf':
        pvalue, ar_lag = adfuller(x=log_or_nothing(series), regression=regression, maxlag=maxlag, autolag=autolag)[1::2]
    elif kwargs['step1_unit_root_model'] == 'pp':
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
    if pvalue >= kwargs.get('step1_unit_root_alpha', 0.01):
        return UnivariateSeriesReductionResult(series, has_kept=True)
    else:
        # Post outlier detection
        odmodel: str = kwargs.get('step1_post_od_model', 'knn')
        if odmodel == 'knn':
            knn = KNNOutlierDetector(w=ar_lag, k=1)   # k=1
            x = scipy.stats.zscore(log_or_nothing(series))
            if knn.has_anomaly(x, kwargs.get('step1_post_od_threshold', 3.0)):
                return UnivariateSeriesReductionResult(series, has_kept=True)
        elif odmodel == 'hotelling':
            outliers = banpei.Hotelling().detect(series, kwargs.get('step1_post_od_threshold', 0.01))
            if len(outliers) > 1:
                return UnivariateSeriesReductionResult(series, has_kept=True)
        else:
            raise ValueError(f"{odmodel} == 'knn' or 'hotelling'")
    return UnivariateSeriesReductionResult(series, has_kept=False)


def ar_based_ad_model(orig_series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if kwargs.get('step1_pre_cv', False):
        if detect_with_cv(orig_series, **kwargs):
            return UnivariateSeriesReductionResult(orig_series, has_kept=False)

    match smoother := kwargs.get('step1_smoother'):
        case 'none':
            series = orig_series
        case 'binner':
            series = smooth_with_binner(orig_series, **kwargs)
        case 'moving_average':
            series = smooth_with_ma(orig_series, **kwargs)
        case _:
            raise ValueError(f"Invalid smoother: '{smoother}'")

    ar_threshold: float = kwargs['step1_ar_anomaly_score_threshold']
    ar = AROutlierDetector(series, maxlag=0)
    ar.fit(
        regression=kwargs['step1_ar_regression'],
        lag=kwargs['step1_ar_lag'],
        ic=kwargs['step1_ar_ic'],
    )
    scores: np.ndarray = ar.anomaly_scores_in_sample()
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"scores must contain only finite values. {scores}")
    outliers, abn_th = AROutlierDetector.detect_by_fitting_dist(scores, threshold=ar_threshold)
    if len(outliers) > 0:
        return UnivariateSeriesReductionResult(
            orig_series, has_kept=True, anomaly_scores=scores, abn_th=abn_th, outliers=outliers)
    return UnivariateSeriesReductionResult(orig_series, has_kept=False, anomaly_scores=scores, abn_th=abn_th)


def hotteling_t2_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if kwargs.get('step1_pre_cv', False):
        if detect_with_cv(series, **kwargs):
            return UnivariateSeriesReductionResult(series, has_kept=False)

    outliers = banpei.Hotelling().detect(series, kwargs.get('step1_hotteling_threshold', 0.01))
    if len(outliers) > 0:
        return UnivariateSeriesReductionResult(series, has_kept=True, outliers=outliers)
    return UnivariateSeriesReductionResult(series, has_kept=False)


def sst_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if kwargs.get('step1_pre_cv', False):
        if detect_with_cv(series, **kwargs):
            return UnivariateSeriesReductionResult(series, has_kept=False)

    sst = banpei.SST(w=len(series)//2)
    change_scores: np.ndarray = sst.detect(scipy.stats.zscore(series), is_lanczos=True)
    change_pts: list[tuple[int, float]] = []
    for i, score in enumerate(change_scores):
        if score >= kwargs.get('step1_sst_threshold'):
            change_pts.append((i, score))
    if len(change_pts) > 0:
        return UnivariateSeriesReductionResult(series, has_kept=True, anomaly_scores=change_scores, outliers=change_pts)
    return UnivariateSeriesReductionResult(series, has_kept=False, anomaly_scores=change_scores)


def discover_changepoint_start_time(scores: np.ndarray, topk: int) -> list[tuple[int, float]]:
    maxidxs = scipy.signal.argrelmax(scores)[0]
    minidxs = scipy.signal.argrelmin(scores)[0]
    if len(maxidxs) == 0:
        return []
    if len(minidxs) == 0:
        minidxs = np.array([0])

    # determine whether maidxs include the last scores index (the newest value)
    lookback_idx: int = max(maxidxs[-1], minidxs[-1])
    if any([scores[i-1] <= scores[i] for i in range(scores.size-1, lookback_idx, -1)]):
        maxidxs = np.append(maxidxs, scores.size-1)

    diff_scores: list[tuple[int, float]] = []
    for maxid in maxidxs:
        last_minid = 0
        for minid in minidxs:
            if minid < maxid:
                last_minid = minid
        diff_scores.append((last_minid, scores[maxid] - scores[last_minid]))
    return sorted(diff_scores, key=lambda t: t[1], reverse=True)[:topk]


def differencial_of_anomaly_score_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if kwargs.get('step1_pre_cv', False):
        if detect_with_cv(series, **kwargs):
            return UnivariateSeriesReductionResult(series, has_kept=False)

    train_series, test_series = np.split(series, 2)

    # Phase 1
    ar_threshold: float = kwargs['step1_ar_anomaly_score_threshold']
    ar = AROutlierDetector(train_series)
    ar.fit(
        regression=kwargs['step1_ar_regression'],
        lag=kwargs['step1_ar_lag'],
        ic=kwargs['step1_ar_ic'],
    )
    scores = ar.anomaly_scores_out_of_sample(test_series)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"scores must contain only finite values. {scores}")
    if np.mean(scores) < 4.0:
        return UnivariateSeriesReductionResult(
            series, has_kept=False, anomaly_scores=scores)

    # outliers, abn_th = ar.detect_by_fitting_dist(scores, threshold=ar_threshold)
    # if len(outliers) == 0:
    #     scores = np.append(np.array([np.NaN]*train_series.size, copy=False), scores)
    #     return UnivariateSeriesReductionResult(
    #         series, has_kept=False, anomaly_scores=scores, abn_th=abn_th)

    changepoints = discover_changepoint_start_time(scores, kwargs['step1_changepoint_topk'])
    changepoints = [(p[0]+train_series.size, p[1]) for p in changepoints]
    return UnivariateSeriesReductionResult(
        series, has_kept=True, anomaly_scores=scores, outliers=changepoints)


def fluxinfer_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    if FluxInferAD(series).detect_anomaly(kwargs['step1_fluxinfer_sigma_threshold']):
        return UnivariateSeriesReductionResult(series, has_kept=True)
    return UnivariateSeriesReductionResult(series, has_kept=False)


def hist_and_stationality_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    bincount = np.histogram(series)[0]
    threshold: float = kwargs['step1_hist_ratio_threshold']

    if len(np.where(bincount == series.size)[0]) > 0:
        # It is normal if all datapoints is in the same bin.
        return UnivariateSeriesReductionResult(series, has_kept=False)

    high_density_bins = np.where(bincount >= series.size * threshold)[0]
    if len(high_density_bins) > 0:
        return UnivariateSeriesReductionResult(series, has_kept=True)

    match kwargs['step1_stationality_test']:
        case 'kpss':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                p_value = kpss(series, regression=kwargs['step1_stationality_test_regression'])[1]
            if p_value <= kwargs['step1_stationality_test_alpha']:
                return UnivariateSeriesReductionResult(series, has_kept=True)
        case 'adf':
            p_value = adfuller(x=series, regression=kwargs['step1_stationality_test_regression'])[1]
            if p_value > kwargs['step1_stationality_test_alpha']:
                return UnivariateSeriesReductionResult(series, has_kept=True)
        case 'combined':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                p_value_kpss = kpss(series, regression=kwargs['step1_stationality_test_regression'])[1]
            has_kept_kpss = p_value_kpss <= kwargs['step1_stationality_test_alpha']
            p_value_adf = adfuller(x=series, regression=kwargs['step1_stationality_test_regression'])[1]
            has_kept_adf = p_value_adf > kwargs['step1_stationality_test_alpha']
            return UnivariateSeriesReductionResult(series, has_kept=(has_kept_kpss & has_kept_adf))
        case _:
            raise ValueError(f"Unknown stationality test {kwargs['step1_stationality_test']}")
    return UnivariateSeriesReductionResult(series, has_kept=False)


def residual_integral_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    # step1: in-sample residuals
    max_rss, max_rss_range = residual_integral_max(series)
    if max_rss < kwargs['step1_residual_integral_threshold']:
        return UnivariateSeriesReductionResult(series, has_kept=False)

    change_start_time = 0
    if kwargs['step1_residual_integral_change_start_point']:
        # step2: detect change start time and out-sample errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            algo: rpt.Binseg = rpt.Binseg(model='normal').fit(series)
        bkp: int = algo.predict(n_bkps=1)[0]
        max_rss, max_rss_range = residual_integral_max(series, bkp=bkp)
        change_start_time: int = max_rss_range[0][0]

    return UnivariateSeriesReductionResult(
        series, has_kept=True, outliers=max_rss_range,
        change_start_point=(change_start_time, series[change_start_time]))


def two_samp_test_model(series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
    alpha: float = kwargs['step1_two_samp_test_alpha']
    train_x, test_x = np.split(series, 2)
    match method := kwargs['step1_two_samp_test_method']:
        case 'ks':
            pval: float = scipy.stats.ks_2samp(train_x, test_x).pvalue
        case 'ad':
            pval: float = scipy.stats.anderson_ksamp([train_x, test_x])[2]
        case 'es':
            pval: float = scipy.stats.epps_singleton_2samp(train_x, test_x)[1]
        case _:
            raise ValueError(f"Unknown two-sample test method {method}")
    if pval <= alpha:
        return UnivariateSeriesReductionResult(series, has_kept=True)
    return UnivariateSeriesReductionResult(series, has_kept=False)


def smooth_with_ma(x: np.ndarray, **kwargs: Any) -> np.ndarray:
    w: int = kwargs.get('step1_ma_window_size', 2)
    return ndimg.uniform_filter1d(input=x, size=w, mode='constant', origin=-(w//2))[:-(w-1)]


def smooth_with_binner(x: np.ndarray, **kwargs: Any) -> np.ndarray:
    """ Smooth time series with binner method.
    """
    w: int = kwargs.get('step1_smoother_binner_window_size', 2)
    smoother = BinnerSmoother(n_knots=int(x.size/w), copy=True)
    smoother.smooth(x)
    return smoother.smooth_data[0]


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
