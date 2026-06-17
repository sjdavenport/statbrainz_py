"""graph_cc (mirrors StatBrainz/Surface/graph_cc.m)."""

import numpy as np

__all__ = ['graph_cc']


def graph_cc(data, thresh, adj_matrix):
    """Connected components of the supra-threshold sub-graph.

    Parameters
    ----------
    data : numpy.ndarray
        Per-vertex statistic.
    thresh : float
        Vertices with ``data > thresh`` survive.
    adj_matrix : scipy.sparse matrix
        Vertex adjacency.

    Returns
    -------
    number_of_clusters : int
    occurences : numpy.ndarray
        Counts per distinct cluster size (sizes > 1).
    cluster_sizes : numpy.ndarray
        Distinct cluster sizes (> 1).
    index_locations : list of numpy.ndarray
        0-based vertex indices per cluster.
    """
    from scipy.sparse.csgraph import connected_components

    data = np.asarray(data).ravel()
    survived = data > thresh
    A = adj_matrix.tolil(copy=True)
    notsurv = np.nonzero(~survived)[0]
    A[notsurv, :] = 0
    A[:, notsurv] = 0
    A = A.tocsr()

    n_comp, labels = connected_components(A, directed=False)
    labels = labels.copy()
    labels[notsurv] = -1  # mark removed vertices

    valid = [lab for lab in np.unique(labels) if lab >= 0]
    index_locations = [np.nonzero(labels == lab)[0] for lab in valid]
    sizes = np.array([len(idx) for idx in index_locations])

    cluster_sizes = np.unique(sizes[sizes > 1]) if sizes.size else np.array([], int)
    occurences = np.array(
        [np.sum(sizes == s) for s in cluster_sizes], dtype=int
    )
    return len(index_locations), occurences, cluster_sizes, index_locations
