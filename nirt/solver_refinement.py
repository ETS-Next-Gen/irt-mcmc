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
                 num_theta_sweeps: int = 2,
                 recorder=None):
        super(SolverRefinement, self).__init__(
            x, item_classification, grid_method=grid_method, improve_theta_method=improve_theta_method,
            num_iterations=num_iterations, num_theta_sweeps=num_theta_sweeps)
        self._recorder = recorder

    def solve(self) -> np.array:
        """Solves the IRT model and returns thetas. Continuation in IRF resolution."""
        logger = logging.getLogger("Solver.solve")

        # Continuation/simulated annealing initialization.
        v = np.ones(self.C, )  # Fixed theta variance in every dimension.

        # Keeping all persons in the active set at all times.
        active = np.arange(self.P, dtype=int)
        person_ind = np.tile(active[:, None], self.C).flatten()
        c_ind = np.tile(np.arange(self.C)[None, :], len(active)).flatten()
        active_ind = (person_ind, c_ind)

        # Prepare a list of IRF resolution levels, from coarsest to finest.
        # coarsest_resolution = 4 bins.
        finest_bin_resolution = self.P // 10
        n = 2 ** np.arange(2, int(np.log2(finest_bin_resolution)+1))

        # Starting from the initial guess (that makes theta approximately standardized), execute continuation steps
        # of increasingly finer IRF resolution.
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        if self._recorder:
            self._recorder.add_theta(0, theta)
        for num_bins in n:
            # Continuation step.
            theta[active] = self._solve_at_resolution(theta[active], v, active_ind, num_bins)
        return theta, v

    def _solve_at_resolution(self, theta_active, v, active_ind, num_bins) -> np.array:
        theta_improver = nirt.theta_improvement.theta_improver_factory(
            self._improve_theta_method, self._num_theta_sweeps)
        for iteration in range(self._num_iterations):
            # Alternate between updating the IRF and improving theta by MLE.
            self.irf = self._update_irf(num_bins, theta_active)
            if self._recorder:
                self._recorder.add_irf(num_bins, self.irf)
            likelihood = nirt.likelihood.Likelihood(self.x, self.c, self.irf)
            theta_active = theta_improver.run(likelihood, theta_active, v, active_ind)
            if self._recorder:
                self._recorder.add_theta(num_bins, theta_active)
        return theta_active
