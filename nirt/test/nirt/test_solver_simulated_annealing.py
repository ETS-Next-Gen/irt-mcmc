import logging
import nirt.simulate.simulate_data as sim
import nirt.solver_simulated_annealing
import numpy as np
import unittest


class TestSolverSimulatedAnnealing(unittest.TestCase):

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

        # Check that the norm of the difference between the approximate and exact IRF at the nodes is small enough for
        # all items, and that this error decreases when the sample size increases.
        num_experiments = 10
        for P in 400 * 4 ** np.arange(4):
            e = [0] * num_experiments
            for experiment in range(num_experiments):
                X, theta, b, c, v = \
                    sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

                # Initial guess for thetas.
                t = nirt.likelihood.initial_guess(X, c)
                # Build IRFs.
                num_bins = 20
                method = "quantile"  # "uniform"
                grid = [nirt.grid.Grid(t[:, ci], num_bins, method=method) for ci in range(C)]
                irf = [nirt.irf.ItemResponseFunction(grid[c[i]], X[:, i]) for i in range(I)]

                max_error = 0
                for i in range(I):
                    # Calculate the scaled, weighted L2 norm of the error in the approximate IRF at the nodes.
                    # Weight = bin count (so that all persons contribute the same weight to the norm: more
                    # dense bins should count more).
                    f = irf[i]
                    exact_irf = sim.three_pl_model(f.node, discrimination, b[i], asym)
                    error = exact_irf - f.probability
                    weight = f.count[f.has_data]
                    error = (sum(weight * error ** 2) / sum(weight)) ** 0.5
                    max_error += error
                max_error /= I
                e[experiment] = max_error
#                print(P, experiment, max_error)
#            print(P, np.mean(e), np.std(e))
            assert np.mean(e) < 0.15

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

        solver = nirt.solver_simulated_annealing.SolverSimulatedAnnealing(X, c)
        theta_approx, _ = solver.solve()

        assert theta_approx.shape == theta.shape

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
        solver = nirt.solver_simulated_annealing.SolverSimulatedAnnealing(X, c)
        theta = solver.solve()
        print(theta)
        assert theta == 0
