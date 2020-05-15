"""The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT.
The person population is down-sampled and gradually up-sampled as IRF resolution is refined."""
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import nirt.solver
import nirt.theta_improvement
import numpy as np


class SolverSampledSimulatedAnnealing(nirt.solver.Solver):
    """The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
    def __init__(self, x: np.array, item_classification: np.array,
                 grid_method: str = "quantile", improve_theta_method: str = "mcmc", num_iterations: int = 3,
                 num_theta_sweeps: int = 10, initial_temperature: int = 1, final_temperature: float = 0.5,
                 initial_sample_per_bin: int = 20):
        super(SolverSampledSimulatedAnnealing, self).__init__(
            x, item_classification, grid_method=grid_method, improve_theta_method=improve_theta_method,
            num_iterations=num_iterations, num_theta_sweeps=num_theta_sweeps)
        self._initial_temperature = initial_temperature
        self._final_temperature = final_temperature
        self._initial_sample_per_bin = initial_sample_per_bin

    def _solve(self, theta) -> np.array:
        """Solves the IRT model and returns thetas. This is the main call that runs an outer loop of simulated
        annealing and an inner loop of IRF building + MCMC sweeps."""
        logger = logging.getLogger("Solver.solve")

        # Continuation/simulated annealing initialization.
        num_bins = 10  # IRF resolution (#bins).
        temperature = self._initial_temperature  # Simulated annealing temperature.
        inactive = np.arange(self.P, dtype=int)
        active = np.array([], dtype=int)
        # An indicator array stating whether each person dimension is currently being estimated. In the current scheme
        # an entire person is estimated (all dimensions) or not (no dimensions), but this supports any set of
        # (person, dimension) pairs.
        likelihood = None
        theta_improver = nirt.theta_improvement.theta_improver_factory(
            self._improve_theta_method, self._num_theta_sweeps, temperature=temperature)

        while temperature >= self._final_temperature:
            # Activate a sample of persons so that the (average) person bin size remains constant during continuation.
            # Note that the temperature may continue decreasing even after all persons have been activated.
            if inactive.size:
                sample_size = self._initial_sample_per_bin * num_bins
                sample = np.random.choice(inactive, size=min(inactive.size, sample_size), replace=False)
                # Initialize newly activated persons' thetas by MLE once we have IRFs and thus a likelihood function.
                if likelihood:
                    theta[sample] = [[likelihood.parameter_mle(p, c, max_iter=self._num_theta_sweeps)
                                      for c in range(self.C)] for p in sample]
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
                theta[active] = theta_improver.run(likelihood, theta[active], active_ind)
            num_bins = min(2 * num_bins, self.P // 10)
            temperature *= 0.5

        return theta
