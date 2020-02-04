"""Non-parametric (binned) Item Response Function (IRF) building from thetas."""
import numpy as np

"""Range of item response function domain."""
M = 10


def histogram(X, bins):
    score = np.array([sum(X[b]) for b in bins])
    count = np.array([len(X[b]) for b in bins])
    return score, count


def create_bins(theta, n):
    j = bin_index(theta, n)
    # Inefficient but good enough for now.
    return np.array([np.where(j == index)[0] for index in range(n)])


def sample_bins(theta, n, sample_size):
    theta_bins = create_bins(theta, n)
    return np.array([np.random.choice(b, sample_size) if len(b) else [] for b in theta_bins])


def bin_index(theta, n):
    h = (2 * M)/n
    j = ((theta + M) / h).astype(int)
    # Anything off to the left is lumped into the left-most bin; similarly for the right boundary.
    return np.minimum(np.maximum(j, 0), n-1)


def theta_range(n):
    h = (2 * M) / n
    return np.linspace(-M + h / 2, M - h / 2, n)
