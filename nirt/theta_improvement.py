"""Functions for improving theta at a given IRF resolution."""
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import numpy as np
from typing import Tuple


def theta_improver_factory(kind, num_iterations, **kwargs):
    """A factory method that returns a theta improver instance of kind 'kind'."""
    if kind == "mle":
        return _MleImprover(num_iterations, loss=kwargs["loss"], alpha=kwargs["alpha"])
    elif kind == "mcmc":
        return _McmcImprover(num_iterations, kwargs["temperature"])
    else:
        raise Exception("Unsupported theta improvement algorithm {}".format(kind))


class _MleImprover:
    """Improves theta by calculating the Maximum Likelihood Estimator (MLE) directly via numerical maximization
    (this is the case of zero temperature in MCMC)."""
    def __init__(self, num_iterations, loss: str = "likelihood", alpha: float = 0.0):
        """
        Improves theta by likelihood maximization. All components are maximized separately as the likelihood is
        separable.

        Args:
            num_iterations: Number of iterations to perform for root search. iterations.
        """
        self._num_iterations = num_iterations
        self._loss = loss
        self._alpha = alpha

    def run(self,
            likelihood: nirt.likelihood.Likelihood,
            theta_active: np.array,
            active_ind: np.array) -> \
            Tuple[np.array, nirt.likelihood.Likelihood]:
        return np.array([likelihood.parameter_mle(p, c, max_iter=self._num_iterations,
                                                  theta_init=theta_active[p][c],
                                                  loss=self._loss,
                                                  alpha=self._alpha)
                         for p, c in zip(*active_ind)]).reshape(theta_active.shape)
        # return np.array([likelihood.parameter_mle(p, c, max_iter=self._num_iterations)
        #                  for p, c in zip(*active_ind)]).reshape(theta_active.shape)


class _McmcImprover:
    """Improves theta by MCMC simulations at a given temperature."""
    def __init__(self, num_iterations, temperature):
        """
        Improves theta by MCMC simulations at a given temperature.

        Args:
            num_iterations: Number of MCMC iterations.
            temperature: simulated annlealing temperature for Metropolis-Hastings steps.
        """
        self._num_iterations = num_iterations
        self._temperature = temperature

    def run(self,
            likelihood: nirt.likelihood.Likelihood,
            theta_active: np.array,
            active_ind: np.array) -> \
            Tuple[np.array, nirt.likelihood.Likelihood]:
        logger = logging.getLogger("Solver.solve_at_resolution")
        # Improve theta estimates by Metropolis sweeps.
        # A vector of size len(theta_active) * C containing all person parameters.
        t = theta_active.flatten()
        energy = likelihood.log_likelihood_term(t, active_ind)
        theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, self._temperature)
        ll = sum(energy)
        logger.info("log-likelihood {:.2f}".format(ll))
        for sweep in range(self._num_iterations):
            ll_old = ll
            t, energy = theta_estimator.estimate(t, active=active_ind, energy=energy)
            ll = sum(energy)
            logger.info("MCMC sweep {:2d} log-likelihood {:.4f} increase {:.2f} accepted {:.2f}%".format(
                sweep, sum(energy), ll - ll_old, 100 * theta_estimator.acceptance_fraction))
        return t.reshape(theta_active.shape) #, likelihood


class _McmcImprover:
    """Improves theta by MCMC simulations at a given temperature."""
    def __init__(self, num_iterations, temperature):
        """
        Improves theta by MCMC simulations at a given temperature.

        Args:
            num_iterations: Number of MCMC iterations.
            temperature: simulated annlealing temperature for Metropolis-Hastings steps.
        """
        self._num_iterations = num_iterations
        self._temperature = temperature

    def run(self,
            likelihood: nirt.likelihood.Likelihood,
            theta_active: np.array,
            active_ind: np.array) -> \
            Tuple[np.array, nirt.likelihood.Likelihood]:
        logger = logging.getLogger("Solver.solve_at_resolution")
        # Improve theta estimates by Metropolis sweeps.
        # A vector of size len(theta_active) * C containing all person parameters.
        t = theta_active.flatten()
        energy = likelihood.log_likelihood_term(t, active_ind)
        theta_estimator = nirt.mcmc.McmcThetaEstimator(likelihood, self._temperature)
        ll = sum(energy)
        logger.info("log-likelihood {:.2f}".format(ll))
        for sweep in range(self._num_iterations):
            ll_old = ll
            t, energy = theta_estimator.estimate(t, active=active_ind, energy=energy)
            ll = sum(energy)
            logger.info("MCMC sweep {:2d} log-likelihood {:.4f} increase {:.2f} accepted {:.2f}%".format(
                sweep, sum(energy), ll - ll_old, 100 * theta_estimator.acceptance_fraction))
        return t.reshape(theta_active.shape) #, likelihood
