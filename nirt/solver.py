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
    def __init__(self, x: np.array, item_classification: np.array, num_sweeps: int = 10, num_iterations: int = 10,
                 initial_sample_per_bin: int = 20, method: str = "quantile", improve_theta_method: str = "mcmc"):
        self.x = x
        self.c = item_classification
        self.P = self.x.shape[0]
        self.num_items = self.x.shape[1]
        self._num_sweeps = num_sweeps
        self._num_iterations = num_iterations
        # Number of item classes. Assumes 'item_classification' contain integers in [0..C-1].
        self.C = max(item_classification) + 1
        self._initial_sample_per_bin = initial_sample_per_bin
        self.irf = None
        self._method = method
        self._improve_theta_method = improve_theta_method

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
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        v = np.var(theta[active], axis=0) ** 2
        likelihood = None
        improve_theta = self._improve_theta_factory(self._improve_theta_method)

        while temperature >= final_temperature:
            # Activate a sample of persons so that the (average) person bin size remains constant during continuation.
            # Note that the temperature may continue decreasing even after all persons have been activated.
            if inactive.size:
                sample_size = self._initial_sample_per_bin * num_bins
                sample = np.random.choice(inactive, size=min(inactive.size, sample_size), replace=False)
                # Initialize newly activated persons' thetas by MLE once we have IRFs and thus a likelihood function.
                if likelihood:
                    theta[sample] = [[likelihood.parameter_mle(p, c, max_iter=5) for c in range(self.C)] for p in sample]
                active = np.concatenate((active, sample))
                inactive = np.setdiff1d(inactive, sample)
                logger.info("Sampled {} persons, total active {}, num_bins {} T {}".format(
                    sample_size, len(active), num_bins, temperature))
            # Index arrays for converting theta[active] into flattened form and back.
            person_ind = np.tile(active[:, None], self.C).flatten()
            c_ind = np.tile(np.arange(self.C)[None, :], len(active)).flatten()
            active_ind = (person_ind, c_ind)
            for iteration in range(self._num_iterations):
                self.irf = self._update_irf(num_bins, theta[active])
                likelihood = nirt.likelihood.Likelihood(self.x, self.c, self.irf)
                theta[active] = improve_theta(likelihood, theta[active], active_ind, temperature=temperature)
                v = np.var(theta[active], axis=0) ** 2
            num_bins = min(2 * num_bins, self.P // 10)
            temperature *= 0.5

        return theta, v

    def _improve_theta_factory(self, kind):
        if kind == "mle":
            return self._improve_theta_by_mcmc
        elif kind == "mcmc":
            return self._improve_theta_by_mle
        else:
            raise Exception("Unsupported theta improvement algorithm {}".format(kind))

    def _improve_theta_by_mle(self,
                               likelihood: nirt.likelihood.Likelihood,
                               theta_active: np.array,
                               v: np.array,
                               active_ind: np.array):
        return np.array([likelihood.parameter_mle(p, c, v, max_iter=5) for p, c in zip(*active_ind)])

    def _improve_theta_by_mcmc(self,
                               likelihood: nirt.likelihood.Likelihood,
                               theta_active: np.array,
                               v: np.array,
                               active_ind: np.array,
                               **kwargs) -> \
            Tuple[np.array, nirt.likelihood.Likelihood]:
        logger = logging.getLogger("Solver.solve_at_resolution")
        # Improve theta estimates by Metropolis sweeps.
        # A vector of size len(theta_active) * C containing all person parameters.
        t = theta_active.flatten()
        energy = likelihood.log_likelihood_term(t, v, active_ind)
        theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, kwargs["temperature"])
        ll = sum(energy)
        logger.info("log-likelihood {:.2f}".format(ll))
        for sweep in range(self._num_sweeps):
            ll_old = ll
            t, energy = theta_estimator.estimate(t, active=active_ind, energy=energy)
            ll = sum(energy)
            logger.info("MCMC sweep {:2d} log-likelihood {:.4f} increase {:.2f} accepted {:.2f}%".format(
                sweep, sum(energy), ll - ll_old, 100 * theta_estimator.acceptance_fraction))
        return t.reshape(theta_active.shape), likelihood

    def _update_irf(self, num_bins: int, theta_active: np.array) -> np.array:
        """
        Builds IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
        Bin persons by theta value into n equal bins (percentiles). Note: theta is a vector of all variables we're
        changing. Reshape it to #active_people x C so we can build separate bin grids for different dimensions.

        Args:
            num_bins:
            theta_active:

        Returns:

        """
        logger = logging.getLogger("Solver.solve_at_resolution")
        # Build IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
        # Bin persons by theta value into n equal bins (percentiles). Note: theta is a vector of all variables we're
        # changing. Reshape it to #active_people x C so we can build separate bin grids for different dimensions.
        grid = [nirt.grid.Grid(theta_active[:, c], num_bins, method=self._method) for c in range(self.C)]
        # for c in range(self.C):
        #     print("c", c, grid[c])
        irf = np.array([nirt.irf.ItemResponseFunction(grid[self.c[i]], self.x[:, i]) for i in range(self.num_items)])
        return irf
