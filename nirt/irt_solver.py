"""The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
import nirt.irf
import numpy as np


class Solver:
    def __init__(self, x, item_classification):
        self.x = x
        self.c = item_classification
        self.P = self.x.shape[0]
        self.I = self.x.shape[1]
        # Number of item classes. Assumes 'item_classification' contain integers in [0..C-1].
        self.C = max(item_classification) + 1

    def initial_guess(self):
        """Returns the initial guess for theta."""
        # Person means for each subscale (dimension): P x C
        x_of_dim = np.array([np.mean(self.x[:, np.where(self.c == d)[0]], axis=1) for d in range(self.C)]).transpose()
        # Population mean and stddev of each dimension.
        population_mean = x_of_dim.mean(axis=0)
        population_std = x_of_dim.std(axis=0)
        return (x_of_dim - population_mean) / population_std

    def solve(self):
        theta = self.initial_guess()
        n = 10
        sample_size = 50
        # For each dimension, bin persons by theta values into n bins so that there are at most sample_size in each bin.
        bins = [nirt.irf.sample_bins(self.theta[:, c], n, sample_size) for c in range(self.C)]
        # Build IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
        irf = np.array([nirt.irf.histogram(self.x[:, i], bins[self.c[i]]) for i in range(self.I)])
