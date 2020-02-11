"""Calculates the likelihood function to maximize (for theta)."""
import nirt.irf
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize

# Smallest allowed argument to log inside log-likelihood computations to avoid negative infinity.
SMALL = 1e-15


class Likelihood:
    """Calculates the likelihood P(theta|X) of theta given (theta) given X. With a uniform prior, this is proportional
    to P(X|theta)."""
    def __init__(self, x, item_classification, irf):
        self._c = item_classification
        self._x = x
        self._irf = irf
        # Create IRF of each item = linear interpolant from bin center values. Use only bins that have values;
        # extend the function to the left with P=0 and to the right with P=1.
        self._num_items = irf.probability.shape[0]
        self.num_bins = irf.probability.shape[1]
        bin_centers = nirt.irf.bin_centers(self.num_bins)
        self._irf_func = [Likelihood._irf_interpolant(bin_centers, irf, i) for i in range(self._num_items)]

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
        Returns the log likelihood terms of person responses (self._x) given theta for an active subset of persons and
        dimensions. This is the sum of the individual person-dimension likelihood.

        Args:
            theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
                entries for the persons and dimensions in the 'active' array.
            active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
                Optional, default: None. If None, all theta values are used.
        Returns:
            log likelihood of person responses (self._x) given theta.
        """
        if active is None:
            active = np.unravel_index(np.arange(theta.size), theta.shape)
        # Active person responses to all items (M x I).
        x = self._x[active[0]]
        # Evaluate the IRF for all active persons and all items first. It's a slight waste but can be vectorized into
        # matrix shape. (Could potentially also vectorize the loop over irf_func entries if we can vectorize IRF
        # interpolation.)
        p = np.array([self._irf_func[i](theta) for i in range(self._num_items)]).transpose()
        y = x * _clipped_log(p) + (1 - x) * _clipped_log(1 - p)
        # Calculate an indicator array of whether item i measures dimension active[1][j]. Thus only items measuring
        # the relevant dimension are taken into account in the log likelihood sum of this active person entry.
        item_measures_dimension = (np.tile(self._c, (active[0].size, 1)) == active[1][:, None])
        return np.sum(y * item_measures_dimension, axis=1)

    def parameter_mle(self, p, c, max_iter=2):
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
        def f(theta_pc): return -self.log_likelihood_term(p, c, theta_pc)
        result = scipy.optimize.minimize_scalar(f, method="brent", options={"maxiter": max_iter})
        # The result struct also contains the function value, which could be useful for further MCMC steps, but
        # for now just returning the root value.
        return result.x

    def plot_irf(self, i):
        plt.figure(1)
        plt.clf()
        # Draw the IRF interpolant.
        t = np.linspace(-nirt.irf.M, nirt.irf.M, 10 * self.num_bins + 1)
        plt.plot(t, self._irf_func[i](t), 'b-')
        # Draw the interpolation nodes (bin centers + extension nodes).
        has_data = self._irf.count[i] > 0
        bin_centers = nirt.irf.bin_centers(self.num_bins)
        x = np.concatenate(([-nirt.irf.M], bin_centers[has_data], [nirt.irf.M]))
        y = np.concatenate(([0], self._irf.probability[i, has_data], [1]))
        plt.plot(x, y, 'ro')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$P(X=1|\theta)$')
        plt.title(r'Item {} response function'.format(i))

    def plot_person_log_likelihood(self, p, c):
        plt.figure(1)
        plt.clf()
        t = np.linspace(-nirt.irf.M, nirt.irf.M, 10 * self.num_bins + 1)
        likelihood = np.array([self.log_likelihood_term(p, c, x) for x in t])
        plt.plot(t, likelihood, 'b-')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\log P(\theta_{pc}|X)$')
        plt.title(r'Log Likelihood person {} dimension {}'.format(p, c))

    @staticmethod
    def _irf_interpolant(bin_centers, irf, i):
        has_data = irf.count[i] > 0
        x = np.concatenate(([-nirt.irf.M], bin_centers[has_data], [nirt.irf.M]))
        y = np.concatenate(([0], irf.probability[i, has_data], [1]))
        return scipy.interpolate.interp1d(x, y, kind="linear", bounds_error=False, fill_value=(0, 1))


def _clipped_log(x):
    return np.log(np.maximum(x, SMALL))
