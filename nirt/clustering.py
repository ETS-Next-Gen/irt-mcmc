import numpy as np
import sklearn.cluster
import sklearn.metrics.pairwise


def abs_cos_dist(x, y=None):
    """
    Returns the absolute cosine distance between x and itself or x and y.

    Args:
        x: {array-like, sparse matrix}, shape (n_samples_1, n_features)
        y: {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Returns:
        disstance: array, shape (n_samples_1, n_samples_2), pairwise cosine distance matrix.
    """
    x = x / np.linalg.norm(x, axis=1)[:, None]
    if y is not None:
        y = y / np.linalg.norm(y, axis=1)[:, None]
    # Once x and y have been normalized, the cos distance can be related to the Euclidean distance, which is fast
    # to calculate n=by scikit-learn.
    return np.abs(1 - 0.5 * sklearn.metrics.pairwise.euclidean_distances(x, Y=y, squared=True))
