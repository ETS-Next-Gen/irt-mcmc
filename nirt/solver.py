"""The main IRT solver that alternates between IRF calculation given theta and theta MCMC estimation given IRT."""
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import nirt.theta_improvement
import numpy as np
from typing import Tuple
from scipy.special import logit


class Solver:
    """Solver base class."""
    def __init__(self, x: np.array, item_classification: np.array,
                 grid_method: str = "quantile", improve_theta_method: str = "mcmc", num_iterations: int = 3,
                 num_theta_sweeps: int = 10, num_smoothing_sweeps: int = 0, theta_init: np.array = None,
                 loss: str = "likelihood", alpha: float = 0.0, use_logit: bool = False):
        self.x = x
        if use_logit:
            self.x = logit(self.x)
        self.c = item_classification
        self.P = self.x.shape[0]
        self.num_items = self.x.shape[1]
        # Number of item classes. Assumes 'item_classification' contain integers in [0..C-1].
        self.C = max(item_classification) + 1
        self.irf = None
        self._method = grid_method
        self._improve_theta_method = improve_theta_method
        self._loss = loss
        self._alpha = alpha
        self._num_iterations = num_iterations
        self._num_theta_sweeps = num_theta_sweeps
        self._num_smoothing_sweeps = num_smoothing_sweeps
        self._use_logit = use_logit

        # Determine the IRF axis for fixed resolutions.
        if theta_init is not None:
            theta = theta_init
        else:
            theta = nirt.likelihood.initial_guess(self.x, self.c)
        theta = (theta - np.mean(theta, axis=0)) / np.std(theta, axis=0)
        self._xlim = [(theta[:, ci].min() - 1, theta[:, ci].max() + 1) for ci in range(self.C)]

    def solve(self) -> np.array:
        """
        Runs the solver.
        Returns: theta estimate.
        """
        logger = logging.getLogger("Solver.solve")
        # Starting from the initial guess (that makes theta approximately standardized), execute continuation steps
        # of increasingly finer IRF resolution.
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        logger.info("Initial guess range [{:.2f}, {:.2f}] mean {:.2f} std {:.2f}".format(
            theta.min(), theta.max(), np.mean(theta, axis=0)[0], np.std(theta, axis=0)[0]))
        return self._solve(theta)

    def _solve(self) -> np.array:
        """
        Runs the solver.
        Returns: theta estimate.
        """
        raise ValueError("Must be implemented by sub-classes")

    def _update_irf(self, num_bins: int, theta_active: np.array, x: np.array = None) -> np.array:
        """
        Builds IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
        Bin persons by theta value into n equal bins (percentiles). Note: theta is a vector of all variables we're
        changing. Reshape it to #active_people x C so we can build separate bin grids for different dimensions.

        Args:
            num_bins:
            theta_active:

        Returns:

        """
        if x is None:
            x = self.x
        logger = logging.getLogger("Solver.solve_at_resolution")
        # Build IRFs from theta values. Assuming the same resolution for all item IRFs, so this is an I x n array.
        # Bin persons by theta value into n equal bins (percentiles). Note: theta is a vector of all variables we're
        # changing. Reshape it to #active_people x C so we can build separate bin grids for different dimensions.
        grid = [nirt.grid.create_grid(theta_active[:, ci], num_bins, method=self._method, xlim=self._xlim[ci])
                for ci in range(self.C)]
        return np.array([nirt.irf.ItemResponseFunction(
            grid[self.c[i]], x[:, i], num_smoothing_sweeps=self._num_smoothing_sweeps,
            boundary="extrapolation" if self._use_logit else "constant") for i in range(x.shape[1])])
