"""Calculates the likelihood function to maximize (for theta)."""
import nirt.irf
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize
from scipy.special import logit

# Smallest allowed argument to log inside log-likelihood computations to avoid negative infinity.
SMALL = 1e-30

# Value to extend an interval negative log likelihood function outside the interval.
_LARGE = 1e8


class Likelihood:
    """Calculates the likelihood P(theta|X) of theta given (theta) given X. With a uniform prior, this is proportional
    to P(X|theta)."""

    def __init__(self, x, item_classification, irf):
        self._c = item_classification
        self._x = x
        self.grid = [irf[np.where(item_classification == c)[0][0]].grid for c in range(max(item_classification) + 1)]
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
        # print("theta", theta)
        # print('active', active)
        p = np.array([irf.interpolant(theta) for irf in self._irf]).transpose()
        # Active person responses to all items (M x I).
        x = self._x[active[0]]
        # print('p', p.shape, 'x', x.shape)
        y = x * _clipped_log(p) + (1 - x) * _clipped_log(1 - p)
        # print('y', y.shape)
        # Calculate an indicator array of whether item i measures dimension active[1][j]. Only the items measuring
        # the relevant dimension are taken into account in the log likelihood sum of this active person entry.
        # print("active[1]", active[1].shape)
        item_measures_dimension = (np.tile(self._c, (active[0].size, 1)) == active[1][:, None])
        # print("item_measures_dimension", item_measures_dimension.shape)
        ll = np.sum(y * item_measures_dimension, axis=1)
        # print("ll", ll)
        # Prior of theta[p, c] = log(N(0, 1))
        prior = - 0.5 * theta ** 2
        return ll + prior

    # def log_likelihood_term_derivative(self, theta, active=None):
    #     """
    #     Returns an array of log likelihoods of person responses (self._x) given theta for an active subset of persons
    #     and dimensions. This is the sum of the individual person-dimension likelihood for each component of theta
    #     (and corresponding active[0], active[1] entries).
    #
    #     Args:
    #         theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
    #             entries for the persons and dimensions in the 'active' array.
    #         active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
    #             Optional, default: None. If None, all theta values are used.
    #     Returns:
    #         array of log likelihood of person responses (self._x) given t for each t in theta.
    #     """
    #     if active is None:
    #         active = np.unravel_index(np.arange(theta.size), theta.shape)
    #     # Evaluate the IRF for all active persons and all items first. It's a slight waste but can be vectorized into
    #     # matrix shape. (Could potentially also vectorize the loop over irf_func entries if we can vectorize IRF
    #     # interpolation.)
    #     # print("theta", theta)
    #     # print('active', active)
    #     p = np.array([irf.interpolant(theta) for irf in self._irf]).transpose()
    #     # Active person responses to all items (M x I).
    #     x = self._x[active[0]]
    #     # print('p', p.shape, 'x', x.shape)
    #     y = x * _clipped_log(p) + (1 - x) * _clipped_log(1 - p)
    #     # print('y', y.shape)
    #     # Calculate an indicator array of whether item i measures dimension active[1][j]. Only the items measuring
    #     # the relevant dimension are taken into account in the log likelihood sum of this active person entry.
    #     # print("active[1]", active[1].shape)
    #     item_measures_dimension = (np.tile(self._c, (active[0].size, 1)) == active[1][:, None])
    #     # print("item_measures_dimension", item_measures_dimension.shape)
    #     ll = np.sum(y * item_measures_dimension, axis=1)
    #     # print("ll", ll)
    #     # Prior of theta[p, c] = log(N(0, 1))
    #     prior = - 2 * theta
    #     return ll + prior

    def parameter_mle(self, p: int, c: int, max_iter: int = 10, display: bool = False,
                      theta_init: float = None, loss: str = "likelihood", alpha: float = 0.0) -> float:
        """
        Returns the Maximum Likelihood Estimator (MLE) of a single parameter theta[p, c] (person's c-dimension
        ability). Uses at most 'max_iter' iterations of Brent's method (bisection bracketing) for likelihood
        maximization.

        Args:
            p: person ID.
            c: latent dimension ID.
            max_iter: maximum number
            display: bool, whether to display root finder error messages.
            theta_init: bool, optional, initial guess.
            loss: type of loss function ("likelihood" | "l2").
            alpha: L2 loss regularization parameter.

        Returns: MLE estimator pf theta[p, c].

        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
        """
        active = (np.array([p]), np.array([c]))
#        def f(theta_pc): return -self.log_likelihood_term(np.array([theta_pc]), active=active)[0]
        if loss == "likelihood":
            def f(theta_pc):
                return -self._total_likelihood_sum_implementation(theta_pc, p, c)
        elif loss == "l2":
            def f(theta_pc):
                return self._l2_loss_sum_implementation(theta_pc, p, c)
        elif loss == "l2_logit":
            def f(theta_pc):
                return self._l2_logit_loss_sum_implementation(theta_pc, p, c)
        else:
            raise Exception("Unsupported loss function {}".format(loss))

        # The likelihood function may be non-concave but has piecewise smooth. Use a root finder in every interval,
        # then find the minimum of all interval minima. Benchmarked to be fast (see debugging_log_likelihood notebook).
        def f_interval(theta_pc, left, right): return f(theta_pc) if left < theta_pc and theta_pc < right else _LARGE
        i = np.where(self._c == c)[0][0]
        x = self._irf[i].x
        if theta_init:
            # Search only within the bin the initial guess is in, and the two neighboring bins.
            # TODO: replace by binary search.
            if theta_init <= x[0]:
                bin = 0
            else:
                bin = np.max(np.where(x < theta_init)[0])
            bins = range(max(0, bin - 1), min(len(x) - 1, bin + 2))
        else:
            # No initial guess supplied, search in all bins.
            bins = range(len(x) - 1)
        interval_min_result = \
            (scipy.optimize.minimize_scalar(f, method="bounded", bounds=(x[j], x[j + 1]), bracket=(x[j], x[j + 1]),
                                            options={"maxiter": max_iter, "disp": display})
             for j in bins)
        # The result struct also contains the function value, which could be useful for further MCMC steps, but
        # for now just returning the root value.
        interval_min_result = list(interval_min_result)
        # print('interval_min_result', interval_min_result)
        # for result in interval_min_result:
        #     print((result.fun, result.x))
        # print(min((result.fun, result.x) for result in interval_min_result))
        return min((result.fun, result.x) for result in interval_min_result)[1]

    def plot_person_log_likelihood(self, ax, p, c):
        # Get x-axis from the grid of one of the items measuring dimension c.
        i = np.where(self._c == c)[0][0]
        irf = self._irf[i]
        x = irf.x
        t = np.linspace(x[0], x[-1], 10 * len(x) + 1)
        likelihood = self.log_likelihood_term(t, active=(np.array([p] * t.size), np.array([c] * t.size)))
        L = irf.interpolant(t)
        ax.plot(t, L, "b-")
        for x in irf.x:
            ax.axvline(x, ymin=min(L), ymax=max(L), marker=".", color="k", markersize=1)

#        ax.plot(self.x, self.y, "ro")
        ax.plot(t, likelihood, "b-")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\log P(\theta_{pc}|X)$")
        ax.set_title(r"Log Likelihood person {} dimension {}".format(p, c))

    def _total_likelihood_sum_implementation(self, t, p, c):
        x = self._x[p]
        L = sum(x[i] * _clipped_log(self._irf[i].interpolant(t)) +
                (1 - x[i]) * _clipped_log(1 - self._irf[i].interpolant(t))
                for i in range(len(x)))
        prior = - 0.5 * t ** 2
        return L + prior

    def _l2_loss_sum_implementation(self, t, p, c):
        x = self._x[p]
        return sum((self._irf[i].interpolant(t) - x[i]) ** 2 for i in range(len(x)))

    def _l2_logit_loss_sum_implementation(self, t, p, c):
        x = self._x[p]
        return sum((self._irf[i].interpolant(t) - x[i]) ** 2 for i in range(len(x)))

    def l2_loss(self, t, use_logit: bool = False):
        """
        Returns the global L2 loss over all students and items. This assumes a uni-dimensional theta (C=1). This is

        Loss = 1/(#students * #items) sum_{students p} sum_{items i} (self._irf[i].interpolant(t[p]) - x[p, i]) ** 2

        where x = student responses.

        Args:
            t: theta approximation.
            use_logit: if True, transforms the IRF and x to logit before taking differences.

        Returns: L2 loss.
        """
        c = 0
        if use_logit:
            return np.mean([(self._irf[i].interpolant(t[p]) - logit(self._x[p, i])) ** 2
                            for p in range(t.shape[0]) for i in range(self._x.shape[1])])
        else:
            return np.mean([(self._irf[i].interpolant(t[p]) - self._x[p, i]) ** 2
                            for p in range(t.shape[0]) for i in range(self._x.shape[1])])


def initial_guess(x, c):
    """Returns the initial guess for theta."""
    # Person means for each sub-scale (dimension): P x C
    C = max(c) + 1
    x_of_dim = np.array([np.mean(x[:, np.where(c == d)[0]], axis=1) for d in range(C)]).transpose()
    # Population mean and stddev of each dimension.
    population_mean = x_of_dim.mean(axis=0)
    population_std = x_of_dim.std(axis=0)
    theta = (x_of_dim - population_mean) / population_std
    # Add random noise.
    #d = theta.max(axis=0) - theta.min(axis=0)
    #theta += 0.001 * d * (2 * np.random.rand(*theta.shape) - 1)
    return theta


def _clipped_log(x):
    return np.log(np.maximum(x, SMALL))
