import logging

import pytest

import nirt.irf
import nirt.mcmc
import nirt.simulate.simulate_data as sim
import nirt.solver
import numpy as np
import unittest


class TestMcmc(unittest.TestCase):

    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

        np.random.seed(0)

        # Number of persons.
        self.P = 100
        # Number of items.
        self.I = 20
        # Number of latent ability dimensions (sub-scales).
        self.C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        self.x, self.theta, self.b, self.c = sim.generate_dichotomous_responses(self.P, self.I, self.C, asymptote=0)

    def test_mcmc_with_indicator_small_temperature_decreases_likelihood(self):
        num_bins = 10  # IRF resolution (#bins).
        sample_size = 60  # 30  # Should be num_bins * (5-10)
        temperature = 0.0001  # Simulated annealing temperature.

        # For each dimension, sample persons and bin them by theta value into n equal bins (percentiles).
        sample = np.random.choice(np.arange(self.P, dtype=int), size=min(self.P, sample_size), replace=False)
        # An indicator array stating whether each person dimension is currently being estimated. In the current scheme
        # an entire person is estimated (all dimensions) or not (no dimensions), but this supports any set of
        # (person, dimension) pairs.
        is_active = np.zeros((self.P, self.C), dtype=bool)
        is_active[sample] = True
        active = np.where(is_active)

        theta = nirt.likelihood.initial_guess(self.x, self.c)
        theta_active = theta[is_active]

        # Build an IRF and a likelihood function.
        t = theta_active.reshape(len(sample), self.C)
        grid = [nirt.grid.Grid(t[:, c], num_bins) for c in range(self.C)]
        irf = [nirt.irf.ItemResponseFunction(grid[self.c[i]], self.x[:, i]) for i in range(self.I)]
        likelihood = nirt.likelihood.Likelihood(self.x, self.c, grid, irf)

        # Run Metropolis sweeps and see if likelihood decreases before arriving at the stationary distribution.
        energy = likelihood.log_likelihood_term(theta_active, active)
        theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, temperature)
        ll = sum(energy)
        num_sweeps = 100
        for sweep in range(num_sweeps):
            ll_old = ll
            theta_active, energy = theta_estimator.estimate(theta_active, active=active, energy=energy)
            ll = sum(energy)
            assert ll > ll_old - 1e-3,\
                "MCMC sweep decreased likelihood from {} to {}".format(ll_old, ll)
        assert theta_estimator.acceptance_fraction == pytest.approx(0.0518, 0.001), \
                "Metropolis acceptance should be {} but was {}".format(0.0588, theta_estimator.acceptance_fraction)