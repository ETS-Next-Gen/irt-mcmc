"""Calculates the likelihood function to maximize (for theta)."""
import nirt.irf
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize

# Smallest allowed log-likelihood value to avoid negative infinities.
SMALL = -100


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

    def log_likelihood(self, theta):
        return sum(self.person_log_likelihood(p, c, theta_p[c])
                   for p, theta_p in enumerate(theta) for c in range(theta.shape[1]))

    def person_log_likelihood(self, p, c, theta_pc):
        terms = [self._irf_func[i](theta_pc) if self._x[p, i] else (1 - self._irf_func[i](theta_pc))
                 for i in range(self._num_items) if self._c[i] == c]
        return SMALL if min(terms) < 1e-15 else sum(map(np.log, terms))

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
        likelihood = np.array([self.person_log_likelihood(p, c, x) for x in t])
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
