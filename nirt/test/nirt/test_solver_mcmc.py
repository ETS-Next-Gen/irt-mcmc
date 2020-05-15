import logging
import nirt.simulate.simulate_data as sim
import nirt.solver_mcmc
import numpy as np
import unittest


class TestSolverMcmc(unittest.TestCase):

    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(0)

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
        X, theta, b, c = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

        solver = nirt.solver_mcmc.SolverMcmc(X, c)
        theta_approx = solver.solve()

        assert theta_approx.shape == theta.shape
