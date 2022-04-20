from typing import Callable, Union

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors


def pearsonr_dist(X, Y, **kwargs):
    r = scipy.stats.pearsonr(X, Y)[0]
    return 1-r


def learn_clusters(
    X: np.ndarray,
    dist_func: Union[str, Callable] = 'pearsonr',
    min_pts: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if dist_func == 'pearsonr':
        dist_func = pearsonr_dist

    if len(X) <= 2:
        # avoid "ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3"
        return np.array([]), np.array([])

    nn_fit = NearestNeighbors(n_neighbors=min_pts, metric=dist_func).fit(X)
    distances = nn_fit.kneighbors()[0]
    dist_square_matrix: scipy.sparse.csr_matrix = nn_fit.radius_neighbors_graph(mode='distance', sort_results=True)

    eps = max(distances.flatten())/4  # see DBSherlock paper

    labels = sklearn.cluster.DBSCAN(
        eps=eps, min_samples=min_pts, metric='precomputed',
    ).fit_predict(dist_square_matrix)

    return labels, dist_square_matrix.toarray()
