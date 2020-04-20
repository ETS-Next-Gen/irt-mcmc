import logging
import nirt.simulate.simulate_data
import nirt.solver
import numpy as np
import unittest


class TestSolver(unittest.TestCase):

    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

    def test_initial_guess(self):
        """Test """
    # For each dimension, bin ALL persons by theta values into n bins so that there are at most
    # sample_size in each bin.
    num_bins = 20
    method = "quantile"  # "uniform"
    t = theta
    grid = [nirt.grid.Grid(t[:, ci], num_bins, method=method) for ci in range(C)]
    irf = [nirt.irf.ItemResponseFunction(grid[c[i]], X[:, i]) for i in range(I)]

    def test_solve_unidimensional_theta(self):
        np.random.seed(0)

        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1
        X, theta, b, c = \
            nirt.simulate.simulate_data.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

        solver = nirt.solver.Solver(X, c)
        theta_approx = solver.solve()

        print(theta)
        assert theta_approx.shape == theta.shape

    @unittest.skip
    def test_solve_multidimensional_theta(self):
        np.random.seed(0)

        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 5
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1
        self.X, self.theta, self.b, self.c = \
            nirt.simulate.simulate_data.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)
        solver = nirt.solver.Solver(self.X, self.c)
        theta = solver.solve()
        print(theta)
        assert theta == 0
