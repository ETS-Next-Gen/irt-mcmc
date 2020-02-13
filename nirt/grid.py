"""An adaptive grid of bins of theta values in one dimension. Instead of a uniform grid, we now use percentiles, to
maximize and balance the sample size per bin."""
import bisect
import numpy as np
import pandas as pd


class Grid:
    """
    An adaptive grid of quantile bins of N theta values in one dimension (sub-scale). By definition, all bins have the
    same size.

    Attributes:
        num_bins: int, number of bins (quantile intervals).
        bin_index: array<int>, shape=(N,) bin index of each person.
        bin: array<array<int>>, shape=(num_bins,) list of bins, each of which is the list of person indices in that
            bin.
        endpoint: array<int>, shape=(N+1,), bin endpoints.
        center: array<int>, shape=(N+1,), bin centers.
    """

    def __init__(self, theta, num_bins):
        """
        Creates a quantile grid.

        Args:
            theta: array<int>, shape=(N,) person latent abiities along a particular dimension.
            num_bins: int, number of bins (quantile intervals).
        """
        assert num_bins >= 3
        self.num_bins = num_bins
        self.bin_index, self.endpoint = pd.qcut(theta, num_bins, retbins=True, labels=np.arange(num_bins, dtype=int))
        self.bin = np.array([np.where(self.bin_index == index)[0] for index in range(num_bins)])
        self.center = 0.5*(self.endpoint[:-1] + self.endpoint[1:])

    def __repr__(self):
        return "Grid[num_bins={}, centers={}]".format(self.num_bins, self.center)
