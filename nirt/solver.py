"""The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import numpy as np
from typing import Tuple


class Solver:
    """The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
    def __init__(self, x: np.array, item_classification: np.array, num_sweeps: int = 10, num_iterations: int = 10):
        self.x = x
        self.c = item_classification
        self.P = self.x.shape[0]
        self.num_items = self.x.shape[1]
        self._num_sweeps = num_sweeps
        self._num_iterations = num_iterations
        # Number of item classes. Assumes 'item_classification' contain integers in [0..C-1].
        self.C = max(item_classification) + 1

    def solve(self) -> np.array:
        """Solves the IRT model and returns thetas. This is the main call that runs an outer loop of simulated
        annealing and an inner loop of IRF building + MCMC sweeps."""
        logger = logging.getLogger("Solver.solve")
        final_temperature = 0.01

        # Continuation/simulated annealing initialization.
        num_bins = 10  # IRF resolution (#bins).
        temperature = 1  # Simulated annealing temperature.
        inactive = np.arange(self.P, dtype=int)
        active = np.array([], dtype=int)
        # An indicator array stating whether each person dimension is currently being estimated. In the current scheme
        # an entire person is estimated (all dimensions) or not (no dimensions), but this supports any set of
        # (person, dimension) pairs.
        is_active = np.zeros((self.P, self.C), dtype=bool)
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        likelihood = None

        while temperature >= final_temperature:
            # Activate a sample of persons so that the (average) person bin size remains constant during continuation.
            # Note that the temperature may continue decreasing even after all persons have been activated.
            if inactive.size:
                sample_size = 5 * num_bins
                sample = np.random.choice(inactive, size=min(inactive.size, sample_size), replace=False)
                is_active[sample] = True
                # Initialize newly activated persons' thetas by MLE once we have IRFs and thus a likelihood function.
                if likelihood:
                    theta[sample] = [[likelihood.parameter_mle(p, c, max_iter=5) for c in range(self.C)] for p in sample]
                active = np.concatenate((active, sample))
                inactive = np.setdiff1d(inactive, sample)
                logger.info("Sampled {} persons, total active {}, num_bins {} T {}".format(
                    sample_size, len(active), num_bins, temperature))
            theta[is_active], likelihood = \
                self._solve_at_resolution(num_bins, temperature, theta[is_active], np.where(is_active))
            num_bins = min(2 * num_bins, self.P // 10)
            temperature *= 0.5

        return theta

    def _solve_at_resolution(self, num_bins: int, temperature: float, theta: np.array, active: np.array) -> \
            Tuple[np.array, nirt.likelihood.Likelihood]:
        logger = logging.getLogger("Solver.solve_at_resolution")
        for iteration in range(self._num_iterations):
            # Build IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
            # Bin persons by theta value into n equal bins (percentiles). Note: theta is a vector of all variables we're
            # changing. Reshape it to #active_people x C so we can build separate bin grids for different dimensions.
            t = theta.reshape(theta.size // self.C, self.C)
            grid = [nirt.grid.Grid(t[:, c], num_bins) for c in range(self.C)]
            irf = [nirt.irf.ItemResponseFunction(grid[self.c[i]], self.x[:, i]) for i in range(self.num_items)]

            # Improve theta estimates by Metropolis sweeps.
            likelihood = nirt.likelihood.Likelihood(self.x, self.c, grid, irf)
            energy = likelihood.log_likelihood_term(theta, active)
            theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, temperature)
            ll = sum(energy)
            logger.info("log-likelihood {:.2f}".format(ll))
            for sweep in range(self._num_sweeps):
                ll_old = ll
                theta, energy = theta_estimator.estimate(theta, active=active, energy=energy)
                ll = sum(energy)
                logger.info("MCMC sweep {:2d} log-likelihood {:.4f} increase {:.2f} accepted {:.2f}%".format(
                    sweep, sum(energy), ll - ll_old, 100 * theta_estimator.acceptance_fraction))
        return theta, likelihood
