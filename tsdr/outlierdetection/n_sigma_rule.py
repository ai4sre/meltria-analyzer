import numpy as np
import scipy.stats


def detect_with_n_sigma_rule(x: np.ndarray, test_start_time: int, sigma_threshold=3) -> np.ndarray:
    """ Detect outliers 'robust z-score' and n-sigma rule.
    """
    train, test = np.split(x, [test_start_time])
    coeff = scipy.stats.norm.ppf(0.75)-scipy.stats.norm.ppf(0.25)
    iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
    niqr = iqr / coeff
    median = np.median(train)
    outlier_idx = np.where(np.abs(test - median) > niqr * sigma_threshold)[0] + test_start_time
    return outlier_idx
