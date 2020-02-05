"""Estimates theta given IRFs using Monte Carlo Markov Chain (MCMC) simulation to approximate the maximum likelihood
estimator for theta."""
import numpy as np
import nirt.irf


class ThetaEstimator:
    def __init__(self, x, irf, temperature):
        self._x = x
        self._irf = irf
        self._temperature = temperature

    @property
    def num_items(self):
        return self.irf.shape[0]

    @property
    def num_bins(self):
        return self.irf.shape[1]

    @property
    def meshsize(self):
        return (2 * nirt.irf.M) / self.num_bins

    def select_proposal(self, theta_p):
        return np.random.normal(theta_p, self.meshsize / 4, 1)

    def log_likelihood(self, theta):
        return sum(self.person_log_likelihood(p, theta_p) for p, theta_p in enumerate(theta))

    def person_log_likelihood(self, p, theta_p):
        j = nirt.irf.bin_index(theta_p, self.num_bins)
        return sum(np.log(self._irf[i, j]) if self._x[p, i] else np.log(1 - self._irf[i, j])
                   for i in range(self.num_items))

    def metropolis(self, p, theta_p):
        theta_proposed = self.select_proposal(theta_p)
        alpha = min(1, np.exp(self.person_log_likelihood(theta_proposed, p) - self.person_log_likelihood(theta_p, p)))
        return theta_proposed if np.random.random() < alpha else theta_p

    def metropolis_sweep(self, theta):
        # Loop implementation may be slow, in which case perhaps Cython can help.
        for p, theta_p in enumerate(theta):
            theta[p] = self.metropolis(p, theta_p)
        return theta