"""Estimates theta given IRFs using Monte Carlo Markov Chain (MCMC) simulation to approximate the maximum likelihood
estimator for theta."""
import logging
import numpy as np
import nirt.irf
import nirt.likelihood
import scipy.optimize


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

    def select_proposal(self, theta_p):
        return np.random.normal(theta_p, self.meshsize / 8, 1)[0]

    def metropolis_step(self, p, c, theta_pc):
        theta_proposed = self.select_proposal(theta_pc)
        energy_proposed = self._likelihood.person_log_likelihood(p, c, theta_proposed)
        energy = self._likelihood.person_log_likelihood(p, c, theta_pc)
        energy_diff = energy_proposed - energy
        alpha = min(1, np.exp(energy_diff / self.temperature))
        accepted = np.random.random() < alpha
        logger = logging.getLogger("metropolis_step")
        logger.debug("p {} theta_p {:.2f} proposed {:.2f} energy_diff {:.2f} = {:.2f} - {:.2f} alpha {:.2f} accepted {"
                 "}".format(
            p, theta_pc, theta_proposed, energy_diff, energy_proposed, energy, alpha, accepted))
        self.num_steps += 1
        self.num_accepted += accepted
        return theta_proposed if accepted else theta_pc

    def estimate(self, theta, active=None):
        """
        Executes a Metropolis-Hastings sweep over all variables. Since we assume each item measures a single dimension
        (sub-scale), we loop over all theta's and all dimensions within a theta (i.e., each theta component is
        separately updated).

        Args:
            theta: array, shape=(M,) active person latent ability parameters. This is a flattened list of all theta
                entries for the persons and dimensions in the 'active' array.
            active: array, shape=(M,) subscripts of active persons and subscales to calculate the likelihood over.
                Optional, default: None. If None, all theta values are used.
        Returns:
             array, shape=(M,) active person parameter values after the sweep.
        """
        if active is None:
            active = np.arange(theta.shape[0], dtype=int)
        # Loop implementation may be slow and can be parallelized. Cython/parallel/numpy vectorization can help.
        return np.array([self.metropolis_step(p, c, theta_pc) for p, c, theta_pc in zip(active[0], active[1], theta)])
