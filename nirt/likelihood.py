"""Calculates the likelihood function to maximize (for theta)."""
import nirt.irf
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize

# Smallest allowed argument to log inside log-likelihood computations to avoid negative infinity.
SMALL = 1e-30


class Likelihood:
    """Calculates the likelihood P(theta|X) of theta given (theta) given X. With a uniform prior, this is proportional
    to P(X|theta)."""

    def __init__(self, x, item_classification, irf):
        self._c = item_classification
        self._x = x
        self._irf = irf

    def log_likelihood(self, theta, active=None):
        """
        Returns the log likelihood of person responses (self._x) given theta for an active subset of persons and
        dimensions. This is the sum of the individual person-dimension likelihood.

        Args:
            theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
                entries for the persons and dimensions in the 'active' array.
            active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
                Optional, default: None. If None, all theta values are used.
        Returns:
            log likelihood of person responses (self._x) given theta.
        """
        return sum(self.log_likelihood_term(theta, active=active))

    def log_likelihood_term(self, theta, active=None):
        """
        Returns an array of log likelihoods of person responses (self._x) given theta for an active subset of persons
        and dimensions. This is the sum of the individual person-dimension likelihood for each component of theta
        (and corresponding active[0], active[1] entries).

        Args:
            theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
                entries for the persons and dimensions in the 'active' array.
            active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
                Optional, default: None. If None, all theta values are used.
        Returns:
            array of log likelihood of person responses (self._x) given t for each t in theta.
        """
        if active is None:
            active = np.unravel_index(np.arange(theta.size), theta.shape)
        # Evaluate the IRF for all active persons and all items first. It's a slight waste but can be vectorized into
        # matrix shape. (Could potentially also vectorize the loop over irf_func entries if we can vectorize IRF
        # interpolation.)
        #print("theta", theta)
        p = np.array([irf.interpolant(theta) for irf in self._irf]).transpose()
        # Active person responses to all items (M x I).
        x = self._x[active[0]]
        #print('p', p.shape, 'x', x.shape)
        y = x * _clipped_log(p) + (1 - x) * _clipped_log(1 - p)
        # print('theta', theta)
        # print('active', active)
        # print('p', p)
        #print('y', y.shape)
        # Calculate an indicator array of whether item i measures dimension active[1][j]. Thus only items measuring
        # the relevant dimension are taken into account in the log likelihood sum of this active person entry.
        #print("active[1]", active[1].shape)
        item_measures_dimension = (np.tile(self._c, (active[0].size, 1)) == active[1][:, None])
        #print("item_measures_dimension", item_measures_dimension.shape)
        ll = np.sum(y * item_measures_dimension, axis=1)
        #print("ll", ll)
        return ll

    def parameter_mle(self, p, c, max_iter=10):
        """
        Returns the Maximum Likelihood Estimator (MLE) of a single parameter theta[p, c] (person's c-dimension
        ability). Uses at most 'max_iter' iterations of Brent's method (bisection bracketing) for likelihood
        maximization.

        Args:
            p: person ID.
            c: latent dimension ID.
            max_iter: maximum number

        Returns: MLE estimator pf theta[p, c].

        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
        """
        active = (np.array([p]), np.array([c]))
        #print("active[0]", active[0].shape)
        #print("active[1]", active[1].shape)
        def f(theta_pc): return -self.log_likelihood_term(np.array([theta_pc]), active=active)[0]
        result = scipy.optimize.minimize_scalar(f, options={"maxiter": max_iter})
        #print(result)
        # The result struct also contains the function value, which could be useful for further MCMC steps, but
        # for now just returning the root value.
        return result.x

    def plot_person_log_likelihood(self, ax, p, c):
        # Get x-axis from the grid of one of the items measuring dimension c.
        i = np.where(self._c == c)[0]
        x = self._irf[i].x
        t = np.linspace(self.x[0], self.x[-1], 10 * len(x) + 1)
        likelihood = self.log_likelihood_term(t, active=(np.array([p] * t.size), np.array([c] * t.size)))
        ax.plot(t, self.interpolant(t), "b-")
        ax.plot(self.x, self.y, "ro")
        plt.plot(t, likelihood, "b-")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\log P(\theta_{pc}|X)$")
        plt.title(r"Log Likelihood person {} dimension {}".format(p, c))


def initial_guess(x, c):
    """Returns the initial guess for theta."""
    # Person means for each sub-scale (dimension): P x C
    C = max(c) + 1
    x_of_dim = np.array([np.mean(x[:, np.where(c == d)[0]], axis=1) for d in range(C)]).transpose()
    # Population mean and stddev of each dimension.
    population_mean = x_of_dim.mean(axis=0)
    population_std = x_of_dim.std(axis=0)
    return (x_of_dim - population_mean) / population_std


def _clipped_log(x):
    return np.log(np.maximum(x, SMALL))
