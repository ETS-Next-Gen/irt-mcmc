"""Estimates theta given IRFs using Joint Maximum Likelihood (MLE)."""
import logging
import numpy as np
import nirt.irf
import scipy.interpolate
import scipy.optimize


class MleThetaEstimator:
    def __init__(self, _likelihood):
        self._likelihood = _likelihood

    def _parameter_mle(self, p, c):
        def f(theta_pc): return -self._likelihood.person_log_likelihood(p, c, theta_pc)
        # TODO(olivne): may want to start the root search from the previous theta_c value.
        print("bracket", f(-nirt.irf.M), f(nirt.irf.M))
        return scipy.optimize.minimize_scalar(f, bracket=(-nirt.irf.M, nirt.irf.M), method="brent",
                                              options={"xtol": 1.48e-08, 'maxiter': 10})

    def estimate(self, theta):
        """
        Returns the maximum likelhood estimate of all thetas. Since we assume each item measures a single dimension
        (sub-scale), we loop over all theta's and all dimensions within a theta (i.e., each theta component is
        separately estimated).

        theta: unused here (needed to conform to an Estimator interface).

        Returns:
             array, shape=(N, C) MLE parameter estimates.
        """
        # Loop implementation may be slow; can instead vectorize function evaluation during maximization.
        return np.array([[self._parameter_mle(p, c) for c in range(theta.shape[1])] for p in range(theta.shape[0])])
