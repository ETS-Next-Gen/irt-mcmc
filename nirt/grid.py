"""An adaptive grid of bins of theta values in one dimension. Instead of a uniform grid, we now use percentiles, to
maximize and balance the sample size per bin."""
import bisect
import numpy as np
from typing import Tuple

"""Limit for fixed uniform domain for theta: [-M,M]."""
M = 8


def create_grid(theta: np.array, num_bins: int, method: str = "quantile", xlim=(-M, M)) -> None:
    """
    A factory method that creates a grid.

    Args:
        theta: array<int>, shape=(N,) person latent abilities along a particular dimension.
        num_bins: int, number of bins in grid.
            bin.
        method: str, optional, default: "quantile". binning strategy.
            "quantile" = equal bins (theta quantiles). IRF has the same accuracy across all thetas, but bins are not
            relatable to the MCMC proposal distribution standard deviation.
            "uniform" = uniform spacing in theta. Meshsize is proportional to the MCMC proposal distribution standard
            deviation, but some bins may be very small, introducing large errors in the IRF estimate.
            "uniform-fixed": uniform over [-M,M] (fixed domain).
     """
    assert num_bins >= 3
    if method == "quantile":
        return _QuantileGrid(theta, num_bins)
    elif method == "uniform" or method == "uniform-fixed":
        if method == "uniform-fixed":
            left, right = xlim
        else:
            left, right = min(theta), max(theta)
        return _UniformGrid(theta, num_bins, left, right)
    else:
        raise ValueError("Unsupported binning strategy '{}'".format(method))


class Grid:
    """
    An adaptive grid of quantile bins of N theta values in one dimension (sub-scale). By definition, all bins have the
    same size.

    Attributes:
        num_bins: int, number of bins (quantile intervals).
        bin: array<array<int>>, shape=(num_bins,) list of bins, each of which is the list of person indices in that
            bin.
        method: str, optional, default: "quantile". binning strategy.
            "quantile" = equal bins (theta quantiles). IRF has the same accuracy across all thetas, but bins are not
            relatable to the MCMC proposal distribution standard deviation.
            "uniform" = uniform spacing in theta. Meshsize is proportional to the MCMC proposal distribution standard
            deviation, but some bins may be very small, introducing large errors in the IRF estimate.
        endpoint: array<int>, shape=(N+1,), bin endpoints.
        center: array<int>, shape=(N+1,), bin centers.
    """
    def __init__(self, num_bins: int) -> None:
        """
        Creates a grid.

        Args:
            theta: array<int>, shape=(N,) person latent abilities along a particular dimension.
            num_bins: int, number of bins (quantile intervals).
        """
        self.num_bins = num_bins
        self._center = None

    def __repr__(self) -> str:
        return "Grid[num_bins={}, center={}, endpoint={}]".format(self.num_bins, self.center, self.endpoint)

    @property
    def range(self) -> Tuple[float, float]:
        """Returns the grid range."""
        return self.endpoint[0], self.endpoint[-1]

    @property
    def center(self) -> np.array:
        """Returns an array of bin centers."""
        if self._center is None:
            self._center = 0.5 * (self.endpoint[:-1] + self.endpoint[1:])
        return self._center

    def update(self, p, theta_p):
        """
        Updates the theta value of person p in the grid.
        Args:
            p: person index.
            theta_p: new theta value.
        Returns: None
        """
        raise ValueError("Must be implemented by sub-classes")


class _UniformGrid(Grid):
    """
    An adaptive grid of quantile bins of N theta values in one dimension (sub-scale). By definition, all bins have the
    same size.

    Attributes:
        num_bins: int, number of bins (quantile intervals).
        bin_index: array<int>, shape=(N,) bin index of each person.
        bin: array<array<int>>, shape=(num_bins,) list of bins, each of which is the list of person indices in that
            bin.
        method: str, optional, default: "quantile". binning strategy.
            "quantile" = equal bins (theta quantiles). IRF has the same accuracy across all thetas, but bins are not
            relatable to the MCMC proposal distribution standard deviation.
            "uniform" = uniform spacing in theta. Meshsize is proportional to the MCMC proposal distribution standard
            deviation, but some bins may be very small, introducing large errors in the IRF estimate.
        endpoint: array<int>, shape=(N+1,), bin endpoints.
        center: array<int>, shape=(N+1,), bin centers.
    """
    def __init__(self, theta: np.array, num_bins: int, left: float, right: float) -> None:
        """
        Creates a uniform grid.

        Args:
            theta: array<int>, shape=(N,) person latent abilities along a particular dimension.
            num_bins: int, number of bins (quantile intervals).
        """
        super(_UniformGrid, self).__init__(num_bins)
        meshsize = (right - left) / num_bins
        bin_index = ((theta - left) // meshsize).astype(int)
        self.endpoint = np.linspace(left, right, num_bins + 1)
        # TODO: speed up this computationally inefficient code.
        self.bin = np.array([np.where(bin_index == index)[0] for index in range(num_bins)])

    def update(self, p, theta_p):
        """A fixed, uniform grid does not depend on the theta values."""
        pass


class _QuantileGrid(Grid):
    """
    An adaptive grid of quantile bins of N theta values in one dimension (sub-scale). By definition, all bins have the
    same size.

    Attributes:
        num_bins: int, number of bins (quantile intervals).
        bin_index: array<int>, shape=(N,) bin index of each person.
        bin: array<array<int>>, shape=(num_bins,) list of bins, each of which is the list of person indices in that
            bin.
        method: str, optional, default: "quantile". binning strategy.
            "quantile" = equal bins (theta quantiles). IRF has the same accuracy across all thetas, but bins are not
            relatable to the MCMC proposal distribution standard deviation.
            "uniform" = uniform spacing in theta. Meshsize is proportional to the MCMC proposal distribution standard
            deviation, but some bins may be very small, introducing large errors in the IRF estimate.
    """
    def __init__(self, theta: np.array, num_bins: int) -> None:
        """
        Creates a quantile grid.

        Args:
            theta: array<int>, shape=(N,) person latent abilities along a particular dimension.
            num_bins: int, number of bins (quantile intervals).
        """
        super(_QuantileGrid, self).__init__(num_bins)
        n = len(theta)

        # Sort theta values in increasing order. Store sorted theta values.
        index = np.argsort(theta)
        self._index = index
        self._theta = theta
        self._sorted_theta = theta[index]

        # Divide the sorted thetas into num_bins as-equal-as-possible ranges.
        bin_start = np.concatenate(([0], np.cumsum([n // num_bins] * (num_bins - n % num_bins) +
                                                  [n // num_bins + 1] * (n % num_bins))))
        self._bin_start = bin_start
        self.bin = [index[bin_start[i]:bin_start[i + 1]] for i in range(num_bins)]
        self.endpoint = np.concatenate((theta[index[bin_start[:-1]]], [theta[index[-1]]]))

    def update(self, p, theta_new):
        """Updates the quantiles and sorted indices when theta[p] is updated to theta_p."""
        # Update sorted theta.
        k = bisect.bisect(self._sorted_theta, self._theta[p])
        np.delete(self._sorted_theta, k)
        k_new = bisect.bisect(self._sorted_theta, theta_new)
        np.insert(self._sorted_theta, k_new, theta_new)

        # Update original theta.
        self._theta[p] = theta_new

        # Update index.
        np.delete(self._index, k)
        np.insert(self._index, k_new, theta_new)

        # Update bins.
        # TODO: instead of this simple call, make this an O(1) operation by only updating the bins
        # in the range of bisect.bisect(self._bin_start, k) - 1, bisect.bisect(self._bin_start, k) - 1
        self.bin = [self._index[self._bin_start[i]:self._bin_start[i + 1]] for i in range(self.num_bins)]

        # b = bisect.bisect(self._bin_start, k) - 1
        # b_new = bisect.bisect(self._bin_start, k) - 1
        # if k < k_new:
        #     np.delete(self._bin[b], k - self._bin_start[b])
        #     self._bin[b].append(self._bin[b + 1][0])
        #     for l in range(b, b_new):
        #         np.delete(self._bin][l], 0)
        #         self._bin[l] = self._bin[l][1:] + [self.bin[l + 1][0]]
