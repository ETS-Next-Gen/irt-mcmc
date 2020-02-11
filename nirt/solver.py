"""The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
import logging
import nirt.irf
import nirt.likelihood
import nirt.mcmc
import numpy as np


class Solver:
    def __init__(self, x: np.array, item_classification: np.array, sample_size: int = 50, num_sweeps: int = 10,
                 num_iterations: int = 10):
        self.x = x
        self.c = item_classification
        self.P = self.x.shape[0]
        self.I = self.x.shape[1]
        self._sample_size = sample_size
        self._num_sweeps = num_sweeps
        self._num_iterations = num_iterations
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
        logger = logging.getLogger("Solver.solve")
        theta = self.initial_guess()
        n = 10 # IRF resolution (#bins).
        temperature = 1 # Simulated annealing temperature.

        # An array of indicators stating whether a person is currently being estimated.
        is_active = np.zeros((self.P, self.C), dtype=bool)
        # For each dimension, bin persons by theta values into n bins so that there are at most sample_size in each bin.
        bins = [nirt.irf.sample_bins(theta[:, c], n, self._sample_size) for c in range(self.C)]
        # Mark persons in the bins as active.
        for c, bin_set in enumerate(bins):
            is_active[np.concatenate(bin_set), c] = True
        logger.info("Sampled persons into bins of minimum size {}; sample size {}".format(
            self._sample_size, sum(len(b) for bin_set in bins for b in bin_set)))
        theta = self.solve_at_resolution(n, temperature, theta, is_active, bins)
        return theta

    def solve_at_resolution(self, n, temperature, theta, is_active, bins):
        logger = logging.getLogger("Solver.solve_at_resolution")
        for iteration in range(self._num_iterations):
            # Build IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
            logger.info("Building IRF")
            irf = nirt.irf.ItemResponseFunction.merge(
                [nirt.irf.histogram(self.x[:, i], bins[self.c[i]]) for i in range(self.I)])
            # Improve theta estimates by Metropolis sweeps / MLE.
            likelihood = nirt.likelihood.Likelihood(self.x, self.c, irf)
            theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, temperature)
            active = np.where(is_active)
            theta_old = theta[is_active]
            logger.info("log-likelikhood {:.2f}".format(likelihood.log_likelihood(theta_old, active)))
            for sweep in range(self._num_sweeps):
                theta[is_active] = theta_estimator.estimate(theta_old, active)
                logger.info("MCMC sweep {:2d} log-likelikhood {:.4f} accepted {:.2f}%".format(
                    sweep, likelihood.log_likelihood(theta[is_active], active),
                    100 * theta_estimator.acceptance_fraction))
        return theta
