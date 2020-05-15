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


class SolverMcmc(nirt.solver.Solver):
    """The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
    def __init__(self, x: np.array, item_classification: np.array,
                 grid_method: str = "quantile", num_iterations: int = 3,
                 num_theta_sweeps: int = 10,
                 coarsest_resolution: int = 4, finest_resolution: int = None,
                 initial_temperature: int = 1, final_temperature: float = 0.5,
                 recorder=None):
        super(SolverMcmc, self).__init__(
            x, item_classification, grid_method=grid_method, improve_theta_method="mcmc",
            num_iterations=num_iterations, num_theta_sweeps=num_theta_sweeps)
        self._initial_temperature = initial_temperature
        self._final_temperature = final_temperature
        self._recorder = recorder
        self._coarsest_resolution = coarsest_resolution
        self._finest_resolution = finest_resolution

    def _solve(self, theta) -> np.array:
        """Solves the IRT model and returns thetas. Continuation in IRF resolution."""
        logger = logging.getLogger("Solver.solve")

        # Keeping all persons in the active set at all times.
        active = np.arange(self.P, dtype=int)
        person_ind = np.tile(active[:, None], self.C).flatten()
        c_ind = np.tile(np.arange(self.C)[None, :], len(active)).flatten()
        active_ind = (person_ind, c_ind)

        # Prepare a list of IRF resolution levels, from coarsest to finest.
        # coarsest_resolution = 4 bins.
        finest_bin_resolution = self._finest_resolution if self._finest_resolution else self.P // 10
        n = [self._coarsest_resolution]
        while n[-1] < finest_bin_resolution:
            n.append(2 * n[-1])
        temperature = self._initial_temperature

        if self._recorder:
            self._recorder.add_theta(0, theta)
        for num_bins in n:
            # Continuation step.
            logger.info("Solving at IRF resolution {}".format(num_bins))
            theta[active] = self._solve_at_resolution(theta[active], active_ind, num_bins, self._initial_temperature)
        return theta

    def _solve_at_resolution(self, theta_active, active_ind, num_bins, temperature) -> np.array:
        logger = logging.getLogger("Solver._solve_at_resolution")
        theta_improver = nirt.theta_improvement.theta_improver_factory(
            self._improve_theta_method, self._num_theta_sweeps, temperature=temperature)
        for iteration in range(self._num_iterations):
            logger.info("Iteration {}/{}".format(iteration + 1, self._num_iterations))
            # Alternate between updating the IRF and improving theta by MLE.
            self.irf = self._update_irf(num_bins, theta_active)
            if self._recorder:
                self._recorder.add_irf(num_bins, self.irf)
            likelihood = nirt.likelihood.Likelihood(self.x, self.c, self.irf)
            theta_active = theta_improver.run(likelihood, theta_active, active_ind)
            theta_active = (theta_active - np.mean(theta_active, axis=0)) / np.std(theta_active, axis=0)
            if self._recorder:
                self._recorder.add_theta(num_bins, theta_active)
        return theta_active
