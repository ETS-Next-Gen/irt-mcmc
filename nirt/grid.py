"""An adaptive grid of bins of theta values in one dimension. Instead of a uniform grid, we now use percentiles, to
maximize and balance the sample size per bin."""
import numpy as np
import pandas as pd
from typing import Tuple


class Grid:
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

    def __init__(self, theta: np.array, num_bins: int, method: str = "quantile") -> None:
        """
        Creates a quantile grid.

        Args:
            theta: array<int>, shape=(N,) person latent abilities along a particular dimension.
            num_bins: int, number of bins (quantile intervals).
        """
        assert num_bins >= 3
        self.num_bins = num_bins
        if method == "quantile":
            self.bin_index, self.endpoint = \
                pd.qcut(theta, num_bins, retbins=True, labels=np.arange(num_bins, dtype=int))
        elif method == "uniform":
            left, right = min(theta) - 1, max(theta) + 1
            meshsize = (right - left) / num_bins
            self.bin_index = ((theta - left) // meshsize).astype(int)
            self.endpoint = np.linspace(left, right, num_bins + 1)
        self.bin = np.array([np.where(self.bin_index == index)[0] for index in range(num_bins)])
        self.center = 0.5 * (self.endpoint[:-1] + self.endpoint[1:])

    def __repr__(self) -> str:
        return "Grid[num_bins={}, centers={}]".format(self.num_bins, self.center)

    @property
    def range(self) -> Tuple[float, float]:
        """Returns the grid range."""
        return self.endpoint[0], self.endpoint[-1]
