"""Estimates theta given IRFs using Monte Carlo Markov Chain (MCMC) simulation to approximate the maximum likelihood
estimator for theta."""
import logging
import numpy as np
import nirt.irf
import nirt.likelihood
from typing import Tuple


class McmcThetaEstimator:
    def __init__(self, likelihood, temperature):
        self._likelihood = likelihood
        self.temperature = temperature
        self.num_accepted = 0
        self.num_steps = 0

    @property
    def acceptance_fraction(self):
        return self.num_accepted / self.num_steps

    @property
    def meshsize(self):
        return (2 * nirt.irf.M) / self._likelihood.num_bins

    def select_proposal(self, theta):
        stddev = self.meshsize / 8
        return np.array([np.random.normal(t, stddev, 1)[0] for t in theta])

    def estimate(self, theta: np.array, active: Tuple[np.array] = None, energy: np.array = None) -> Tuple[np.array]:
        """
        Executes a Metropolis-Hastings sweep over all variables. Since we assume each item measures a single dimension
        (sub-scale), we loop over all theta's and all dimensions within a theta (i.e., each theta component is
        separately updated).

        Args:
            theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
                entries for the persons and dimensions in the 'active' array.
            active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
                Optional, default: None. If None, all theta values are used.
            energy: array, shape=(M,) array of person likelihoods fo all values in theta.
                Optional, default: None. If None, it is recomputed inside this function.
        Returns:
             theta_new: array, shape=(M,) active person parameter values after the sweep.
             energy_new: array, shape=(M,) array of person likelihoods fo all values in theta.
        """
        # Since the likelihood is separable, all theta entry updates are done in parallel using numpy vectorization.
        num_persons = theta.shape[0]
        if active is None:
            active = np.arange(num_persons, dtype=int)
        theta_proposed = self.select_proposal(theta)
        energy_proposed = self._likelihood.person_log_likelihood(theta_proposed, active)
        if energy is None:
            energy = self._likelihood.person_log_likelihood(theta)
        energy_diff = energy_proposed - energy
        alpha = np.minimum(1, np.exp(energy_diff / self.temperature))
        accepted = np.random.random(num_persons) < alpha
        logger = logging.getLogger("metropolis_step")
        if logger.level == logging.DEBUG:
            for p, c, t, t_proposed, e, e_proposed, de, a in zip(active[0], active[1], theta, theta_proposed, energy,
                                                              energy_proposed, energy_diff, accepted):
                logger.debug("p {} theta_p {:.2f} proposed {:.2f} energy_diff {:.2f} = {:.2f} - {:.2f} alpha {:.2f} "
                             "accepted {}".format(p, t, t_proposed, de, e_proposed, e, alpha, accepted))
        self.num_steps += theta.size
        self.num_accepted += sum(accepted)
        theta[accepted] = theta_proposed[accepted]
        energy[accepted] = energy_proposed[accepted]
        return theta, energy
