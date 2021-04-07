"""The main IRT solver that employs a continuation method in the number of items (clusters items, starts from one
item cluster, then 2, etc., till the finest level of items is reached. At each level, we alternate between IRF
# calculation given theta and theta estimation."""
import cntree.cntree as cntree
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import nirt.solver
import numpy as np


class SolverItemClustering(nirt.solver.Solver):
    """The main IRT solver that alternates between IRF calculation given theta and theta estimation given IRT.
    Iterative refinement of the IRF resolution, keeping everything else fixed (using all persons at each refinement
    level)."""
    def __init__(self, x: np.array, item_classification: np.array,
                 grid_method: str = "uniform-fixed", improve_theta_method: str = "mle", num_iterations: int = 2,
                 num_theta_sweeps: int = 2, coarsest_resolution: int = 4, finest_resolution: int = None,
                 recorder=None, num_smoothing_sweeps: int = 0, theta_init: np.array = None,
                 loss: str = "likelihood", alpha: float = 0.0, refine_irf: bool = False, use_logit: bool = False):
        super(SolverItemClustering, self).__init__(
            x, item_classification, grid_method=grid_method, improve_theta_method=improve_theta_method,
            num_iterations=num_iterations, num_theta_sweeps=num_theta_sweeps,
            num_smoothing_sweeps=num_smoothing_sweeps, theta_init=theta_init, loss=loss, alpha=alpha,
            use_logit=use_logit)
        self._recorder = recorder
        self._coarsest_resolution = coarsest_resolution
        self._finest_resolution = finest_resolution
        self._refine_irf = refine_irf

        # Hierarchically cluster items. Use feature vectors = mean success of student quantiles on the item.
        P, I = x.shape
        bin_size = P // 50
        x_coarse = x[nirt.irf.sorted_chunks(x, bin_size)].mean(1)
        tree = cntree.CNTree(debug=1)
        clustering = tree.cluster(x_coarse.transpose())
        # Save all clustering level item labelling in a matrix of size {num_levels} x {num_items}. Save all levels
        # from clustering + finest level = original items.
        self._clustering = [[level.cluster_members(i) for i in range(level.size)]
                            for level in cntree.get_clustering_levels(clustering) if level.size <= 0.5 * I] + \
                           [[np.array([i]) for i in range(I)]]

    def _solve(self, theta) -> np.array:
        """Solves the IRT model and returns thetas. Continuation in IRF resolution."""
        logger = logging.getLogger("Solver.solve")

        # Keeping all persons in the active set at all times.
        active = np.arange(self.P, dtype=int)
        person_ind = np.tile(active[:, None], self.C).flatten()
        c_ind = np.tile(np.arange(self.C)[None, :], len(active)).flatten()
        active_ind = (person_ind, c_ind)

        if self._recorder:
            self._recorder.add_theta(len(self._clustering[0]), theta)
        num_bins = self._coarsest_resolution

        # Continuation in item clustering level (coarse to fine).
        for level in self._clustering:
            logger.info("Solving at item clustering level with {} items {} bins".format(len(level), num_bins))
            x = np.array([self.x[:, cluster].mean(1) for cluster in level]).transpose()
            theta[active] = self._solve_at_resolution(x, theta[active], active_ind, num_bins)
            if self._refine_irf:
                num_bins *= 2
        return theta

    def _solve_at_resolution(self, x, theta_active, active_ind, num_bins) -> np.array:
        num_items = x.shape[1]
        logger = logging.getLogger("Solver._solve_at_resolution")
        theta_improver = nirt.theta_improvement.theta_improver_factory(
            self._improve_theta_method, self._num_theta_sweeps, loss=self._loss, alpha=self._alpha)
        for iteration in range(self._num_iterations):
            logger.info("Iteration {}/{}".format(iteration + 1, self._num_iterations))
            # Alternate between updating the IRF and improving theta by MLE.
            self.irf = self._update_irf(num_bins, theta_active, x=x)
            if self._recorder:
                self._recorder.add_irf(num_items, self.irf)
            likelihood = nirt.likelihood.Likelihood(x, self.c, self.irf)
            theta_active = theta_improver.run(likelihood, theta_active, active_ind)
            theta_active = (theta_active - np.mean(theta_active, axis=0)) / np.std(theta_active, axis=0)
            if self._recorder:
                self._recorder.add_theta(num_items, theta_active)
        return theta_active
