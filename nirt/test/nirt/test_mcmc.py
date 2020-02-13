import logging
import nirt.irf
import nirt.mcmc
import nirt.simulate.simulate_data
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
        self.C = 5
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        self.x, self.theta, self.b, self.c = \
            nirt.simulate.simulate_data.generate_simulated_data(self.P, self.I, self.C, asym=0, discrimination=1)

    def test_mcmc_with_indicator_small_temperature_decreases_likelihood(self):
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        n = 10  # IRF resolution (#bins).
        sample_size = 20
        temperature = 0.01  # Simulated annealing temperature.

        # For each dimension, bin persons by theta values into n bins so that there are at most sample_size in each bin.
        bins = [nirt.irf.sample_bins(theta[:, c], n, sample_size) for c in range(self.C)]

        # An array of indicators stating whether a person is currently being estimated.
        is_active = np.zeros((self.P, self.C), dtype=bool)
        for c, bin_set in enumerate(bins):
            is_active[np.concatenate(bin_set), c] = True
        active = np.where(is_active)
        theta = self.theta[is_active]

        irf = nirt.irf.ItemResponseFunction.merge(
            [nirt.irf.histogram(self.x[:, i], bins[self.c[i]]) for i in range(self.I)])
        likelihood = nirt.likelihood.Likelihood(self.x, self.c, irf)

        # Run Metropolis sweeps and see if likelihood decreases before arriving at the stationary distribution.
        energy = likelihood.log_likelihood_term(theta, active)
        theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, temperature)
        likelihood = sum(energy)
        num_sweeps = 10
        for sweep in range(num_sweeps):
            likelihood_old = likelihood
            theta, energy = theta_estimator.estimate(theta, active=active, energy=energy)
            likelihood = sum(energy)
            assert likelihood > likelihood_old, \
                "MCMC sweep decreased likelihood from {} to {}".format(likelihood_old, likelihood)
            assert 0.5 < theta_estimator.acceptance_fraction < 0.7, \
                "Metropolis acceptance should be around 0.5 but was {}".format(theta_estimator.acceptance_fraction)
