import logging
import nirt.irf
import nirt.grid
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
        self.P = 1000
        # Number of items.
        self.I = 20
        # Number of latent ability dimensions (sub-scales).
        self.C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        self.x, self.theta, self.b, self.c = \
            nirt.simulate.simulate_data.generate_simulated_data(self.P, self.I, self.C, asym=0, discrimination=1)

    def test_parameter_mle_maximizes_likelihood(self):
        # Build an IRF from some reasonable theta values.
        num_bins = 10
        sample_size = 20
        theta = nirt.likelihood.initial_guess(self.x, self.c)
        c = 0
        sample = np.random.choice(np.arange(self.P, dtype=int), size=sample_size * num_bins, replace=False)
        grid = [nirt.grid.Grid(theta[sample, c], num_bins) for c in range(self.C)]
        irf = [nirt.irf.ItemResponseFunction(grid[self.c[i]], self.x[:, i]) for i in range(self.I)]

        likelihood = nirt.likelihood.Likelihood(self.x, self.c, grid, irf)

        for p in range(self.P):
            i = np.where(self.c == c)[0][0]
            x = irf[i].x
            t = np.linspace(x[0], x[-1], 10 * len(x) + 1)
            active = np.tile([p, c], (len(t), 1))
            likelihood_values = likelihood.log_likelihood_term(t, active=(active[:, 0], active[:, 1]))
            print('p', p, theta.shape, active.shape)
            print(likelihood_values.shape)

            # Verify that we can find the LL maximum with a root finder.
            t_mle = likelihood.parameter_mle(p, c, max_iter=2)
            likelihood_mle = likelihood.log_likelihood_term(t_mle, (np.array([p]), np.array([c])))[0]
            print(t_mle, likelihood_mle)
            assert likelihood_mle > max(likelihood_values) - 2, "MLE likelihood {} < max likelihood value on a grid {}".format(
                likelihood_mle, max(likelihood_values))