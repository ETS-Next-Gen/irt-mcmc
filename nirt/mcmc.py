"""Estimates theta given IRFs using Monte Carlo Markov Chain (MCMC) simulation to approximate the maximum likelihood
estimator for theta."""
import logging
import numpy as np
import nirt.likelihood
from typing import Tuple


class McmcThetaEstimator:
    """Estimates theta given IRFs using Monte Carlo Markov Chain (MCMC) simulation to approximate the maximum likelihood
    estimator for theta"""
    def __init__(self, likelihood: nirt.likelihood.Likelihood, temperature: float) -> None:
        self._likelihood = likelihood
        self.temperature = temperature
        # Standard deviation of proposal steps = average bin size in that dimension.
        self._proposal_std = np.array([(g.range[1] - g.range[0]) / g.num_bins for g in likelihood.grid])
        self.num_accepted = 0
        self.num_steps = 0

    @property
    def acceptance_fraction(self) -> float:
        """Returns the Metropolis acceptance fraction over all steps performed to date."""
        return self.num_accepted / self.num_steps

    def estimate(self, theta: np.array, v: np.array, active: Tuple[np.array] = None, energy: np.array = None) -> \
            Tuple[np.array, np.array]:
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
        if active is None:
            active = np.unravel_index(np.arange(theta.size), theta.shape)
        theta_proposed = self._select_proposal(theta, active[1])
        energy_proposed = self._likelihood.log_likelihood_term(theta_proposed, v, active)
        if energy is None:
            energy = self._likelihood.log_likelihood_term(theta, v)
        energy_diff = energy_proposed - energy
        alpha = np.minimum(1, np.exp(np.minimum(500, energy_diff / self.temperature)))
        accepted = np.random.random(len(theta)) < alpha
        logger = logging.getLogger("metropolis_step")
        if logger.level >= logging.DEBUG:
            # noinspection PyTypeChecker
            for p, c, t, t_proposed, e, e_proposed, de, a in \
                    zip(active[0], active[1], theta, theta_proposed, energy, energy_proposed, energy_diff, accepted):
                logger.debug("p {} theta_p {:.2f} proposed {:.2f} energy_diff {:.2f} = {:.2f} - {:.2f} alpha {:.2f} "
                             "accepted {}".format(p, t, t_proposed, de, e_proposed, e, alpha, accepted))
        self.num_steps += theta.size
        # noinspection PyTypeChecker
        self.num_accepted += sum(accepted)
        theta[accepted] = theta_proposed[accepted]
        energy[accepted] = energy_proposed[accepted]
        return theta, energy

    def _select_proposal(self, theta: np.array, dimension: np.array) -> np.array:
        return np.array([np.random.normal(t, self._proposal_std[dim], 1)[0] for t, dim in zip(theta, dimension)])
