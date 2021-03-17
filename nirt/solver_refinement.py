"""The main IRT solver that alternates between IRF calculation given theta and theta estimation. This is an iterative
IRF resolution refinement strategy. All thetas are updated at each resolution, using MLE calculation."""
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import nirt.solver
import numpy as np


class SolverRefinement(nirt.solver.Solver):
    """The main IRT solver that alternates between IRF calculation given theta and theta estimation given IRT.
    Iterative refinement of the IRF resolution, keeping everything else fixed (using all persons at each refinement
    level)."""
    def __init__(self, x: np.array, item_classification: np.array,
                 grid_method: str = "uniform-fixed", improve_theta_method: str = "mle", num_iterations: int = 2,
                 num_theta_sweeps: int = 2, coarsest_resolution: int = 4, finest_resolution: int = None,
                 recorder=None, num_smoothing_sweeps: int = 0, theta_init: np.array = None,
                 loss: str = "likelihood", alpha: float = 0.0):
        super(SolverRefinement, self).__init__(
            x, item_classification, grid_method=grid_method, improve_theta_method=improve_theta_method,
            num_iterations=num_iterations, num_theta_sweeps=num_theta_sweeps,
            num_smoothing_sweeps=num_smoothing_sweeps, theta_init=theta_init, loss=loss, alpha=alpha)
        self._recorder = recorder
        self._coarsest_resolution = coarsest_resolution
        self._finest_resolution = finest_resolution

    def _solve(self, theta) -> np.array:
        """Solves the IRT model and returns thetas. Continuation in IRF resolution."""
        logger = logging.getLogger("Solver.solve")

        # Continuation/simulated annealing initialization.

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

        if self._recorder:
            self._recorder.add_theta(0, theta)
        for num_bins in n:
            # Continuation step.
            logger.info("Solving at IRF resolution {}".format(num_bins))
            theta[active] = self._solve_at_resolution(theta[active], active_ind, num_bins)
        return theta

    def _solve_at_resolution(self, theta_active, active_ind, num_bins) -> np.array:
        logger = logging.getLogger("Solver._solve_at_resolution")
        theta_improver = nirt.theta_improvement.theta_improver_factory(
            self._improve_theta_method, self._num_theta_sweeps, loss=self._loss, alpha=self._alpha)
        for iteration in range(self._num_iterations):
            logger.info("Iteration {}/{}".format(iteration + 1, self._num_iterations))
            # Alternate between updating the IRF and improving theta by MLE.
            self.irf = self._update_irf(num_bins, theta_active)
            if self._recorder:
                self._recorder.add_irf(num_bins, self.irf)
            likelihood = nirt.likelihood.Likelihood(self.x, self.c, self.irf)
            theta_active = theta_improver.run(likelihood, theta_active, active_ind)
            #theta_active = (theta_active - np.mean(theta_active, axis=0)) / np.std(theta_active, axis=0)
            if self._recorder:
                self._recorder.add_theta(num_bins, theta_active)
        return theta_active
