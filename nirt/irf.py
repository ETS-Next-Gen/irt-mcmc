"""Builds non-parametric (binned) Item Response Function (IRF) from thetas."""
import numpy as np

"""Range of item response function domain."""
M = 10


def histogram(x, bins):
    score = np.array([sum(x[b]) for b in bins])
    count = np.array([len(x[b]) for b in bins])
    return score, count


def create_bins(theta, n):
    j = bin_index(theta, n)
    # Inefficient implementation, but good enough for now.
    return np.array([np.where(j == index)[0] for index in range(n)])


def sample_bins(theta, n, sample_size):
    theta_bins = create_bins(theta, n)
    return np.array([np.random.choice(b, sample_size) if len(b) else [] for b in theta_bins])


def bin_index(theta, n):
    """
    Returns the bin index of a latent ability value 'theta'. If |theta| <= M, bins are uniformly spaced. Anything off
    to the left is lumped into the left-most bin; similarly for the right boundary.

    @param theta: person latent ability in a certain dimension.
    @param n: number of bins
    @return: bin index in [0..j-1].
    """
    theta_left, h = -M, (2 * M) / n
    return np.minimum(np.maximum(((theta - theta_left) / h).astype(int), 0), n - 1)


def bin_centers(n):
    """
    Returns an array of bin centers.
    @param n: number of bins.
    @return: array, shape=(n,) bin centers.
    """
    h = (2 * M) / n
    return np.linspace(-M + h / 2, M - h / 2, n)
