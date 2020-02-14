# irt

## Monte-Carlo Markov Chain (MCMC) for Item Response Theory (IRT) Models

This is a framework for fast estimation of both student latent abilities (thetas) and Item Respons Functions (IRFs) using MCMC within a continuation/simulated annealing schedule.
we
This is a non-parametric IRT (NIRT) model: we alternate between constructing IRFs by binning thetas and creating a score histogram; and MCMC, to improve the theta estimate. These iterations are performed within an outer loop in which the MCMC temperature (used in Metropolis-Hastings steps) is decreased while increasing IRF resolution.

In the general case, we also estimate the number of dimensions C of theta by clustering items. IRFs are defined over the C dimensional step. theta represents latent ability dimensions (e.g., spatial thinking, deduction), which are not the same as the reported subscales (e.g., algebra, geometry).

Applications to IRT models.

Contact/maintainer: Oren Livne (olivne@ets.org)

Licensing: License TBD.