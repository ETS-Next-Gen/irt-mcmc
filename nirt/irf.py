"""Builds non-parametric (binned) Item Response Function (IRF) from thetas."""
import numpy as np

"""Range of item response function domain."""
M = 10


class ItemResponseFunction:
    def __init__(self, score, count):
        self.score = score
        self.count = count

    @property
    def probability(self):
        return self.score / np.maximum(1e-15, self.count)

    def __repr__(self):
        return "count " + repr(self.count) + "\n"
        + "score " + repr(self.score) + "\n"
        + "P " + repr(self.probability) + "\n"

    @staticmethod
    def merge(irf_list):
        return ItemResponseFunction(np.array([irf.score for irf in irf_list]),
                                    np.array([irf.count for irf in irf_list]))

    def __getitem__(self, i):
        return ItemResponseFunction(self.score[i], self.count[i])

def histogram(x, bins):
    score = np.array([sum(x[b]) for b in bins])
    count = np.array([len(x[b]) for b in bins])
    return ItemResponseFunction(score, count)


def create_bins(theta, n):
    j = bin_index(theta, n)
    # Inefficient implementation, but good enough for now.
    return np.array([np.where(j == index)[0] for index in range(n)])


def sample_bins(theta, n, sample_size):
    theta_bins = create_bins(theta, n)
    return np.array([np.random.choice(b, sample_size) if len(b) else np.array([], dtype=np.int64) for b in theta_bins])


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


def _augment(array):
    if array.ndim == 1:
        return array[None, :]
    return array


def _merge(a, b):
    np.concatenate(_augment(a), _augment(b))
