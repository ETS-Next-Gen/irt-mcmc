import logging
import nirt.irf
import nirt.mcmc
import nirt.simulate.simulate_data
import nirt.solver
import numpy as np
import unittest


class TestLikelihood(unittest.TestCase):

    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

        np.random.seed(0)

        # Number of persons.
        self.P = 10
        # Number of items.
        self.I = 20
        # Number of latent ability dimensions (sub-scales).
        self.C = 5
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        self.x, self.theta, self.b, self.c = \
            nirt.simulate.simulate_data.generate_simulated_data(self.P, self.I, self.C, asym=0, discrimination=1)

    def test_parameter_mle_maximizes_likelihood(self):
        # Build an IRF from some reasonable theta values.
        num_bins = 10
        sample_size = 20
        theta = self._initial_guess()
        bins = nirt.irf.sample_bins(theta[:, 0], num_bins, sample_size)
        irf = nirt.irf.ItemResponseFunction.merge([nirt.irf.histogram(self.x[:, i], bins) for i in range(self.I)])

        likelihood = nirt.likelihood.Likelihood(self.x, self.c, irf)
        for p in [1]: #range(self.P):
            c = 1 # np.random.choice(np.arange(self.C), 1)[0]
            # See that we can find this minimum with a root finder.
            t = likelihood.parameter_mle(p, c)

            grid = np.linspace(-nirt.irf.M, nirt.irf.M, 10 * num_bins + 1)
            active = np.tile([p, c], (grid.size, 1))
            likelihood_values = likelihood.log_likelihood_term(grid, active=(active[:, 0], active[:, 1]))
            mle = likelihood.log_likelihood_term(t, (np.array([p]), np.array([c])))[0]
            print(list(grid))
            print(list(likelihood_values))
            print(t, mle)
            assert mle > max(likelihood_values) - 2, "MLE likelihood {} < max likelihood value on a grid {}".format(
                mle, max(likelihood_values))

    def _initial_guess(self):
        """Returns the initial guess for theta."""
        # Person means for each subscale (dimension): P x C
        x_of_dim = np.array([np.mean(self.x[:, np.where(self.c == d)[0]], axis=1) for d in range(self.C)]).transpose()
        # Population mean and stddev of each dimension.
        population_mean = x_of_dim.mean(axis=0)
        population_std = x_of_dim.std(axis=0)
        return (x_of_dim - population_mean) / population_std
