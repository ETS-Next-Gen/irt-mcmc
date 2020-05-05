import logging
import nirt.error
import nirt.run_recorder
import nirt.simulate.simulate_data as sim
import nirt.solver_refinement
import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal


class TestSolver(unittest.TestCase):

    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(0)

    def test_initial_guess(self):
        """Verifies that the histograms built from the initial theta guess approximate the exact IRFs to a reasonable
        degree.
        """
        I = 20
        C = 1
        asym = 0
        discrimination = 1
        # Exact IRFs.
        model_irf = [lambda t, i=i: nirt.simulate.simulate_data.three_pl_model(t, discrimination, b[i], asym)
                     for i in range(I)]

        num_bins = 5
        method = "uniform-fixed" #"quantile"  # "uniform"

        # Check that the norm of the difference between the approximate and exact IRF at the nodes is small enough for
        # all items, and that this error decreases when the sample size increases.
        num_p = 5
        num_experiments = 10
        e_mean = [0] * num_p
        for k, P in enumerate(100 * 4 ** np.arange(num_p)):
            e = [0] * num_experiments
            for experiment in range(num_experiments):
                X, theta, b, c, v = \
                    sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

                # Initial guess for thetas.
                t = nirt.likelihood.initial_guess(X, c)
                # Build IRFs.
                xlim = [(min(t[:, ci]) - 2, max(t[:, ci]) + 2) for ci in range(C)]
                grid = [nirt.grid.Grid(t[:, ci], num_bins, method=method, xlim=xlim[ci]) for ci in range(C)]
                irf = [nirt.irf.ItemResponseFunction(grid[c[i]], X[:, i]) for i in range(I)]

                # Calculate the scaled, weighted L2 norm of the error in the approximate IRF at the nodes.
                # Weight = bin count (so that all persons contribute the same weight to the norm: more
                # dense bins should count more).
                error = nirt.error.error_norm_by_item(model_irf, irf)
                #print("P {} {:.3f} +- {:.3f}".format(P, error.mean(), error.std()))
                e[experiment] = error.mean()
            e_mean[k] = np.mean(e)
        assert_array_almost_equal(e_mean, [
            0.06139,
            0.03444,
            0.02465,
            0.01991,
            0.01670], 5)

    def test_solve_unidimensional_theta(self):
        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1
        X, theta, b, c, v = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

        solver = nirt.solver_refinement.SolverRefinement(X, c)
        theta_approx, _ = solver.solve()

        assert theta_approx.shape == theta.shape

    def test_solve_with_recorder(self):
        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1
        X, theta, b, c, v = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

        recorder = nirt.run_recorder.RunRecorder()
        solver = nirt.solver_refinement.SolverRefinement(X, c, recorder=recorder)
        theta_approx, _ = solver.solve()

        assert recorder.theta.keys() == set([0, 4, 8]), \
            "unexpected recorder theta key set {}".format(recorder.theta.keys())
        assert len(recorder.theta[0]) == 1
        assert all(len(recorder.theta[r]) == 2 for r in recorder.theta.keys() if r != 0)

        assert recorder.irf.keys() == set([4, 8]), \
            "unexpected recorder IRF key set {}".format(recorder.irf.keys())
        assert all(len(recorder.irf[r]) == 2 for r in recorder.irf.keys() if r != 0)

    @unittest.skip
    def test_solve_multidimensional_theta(self):
        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 5
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1
        X, theta, b, c, v = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)
        solver = nirt.solver_refinement.SolverRefinement(X, c)
        theta = solver.solve()
        print(theta)
        assert theta == 0
