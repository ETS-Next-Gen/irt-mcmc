"""Runs the IRT MCMC solver, version 1 on simulate data: each sub-scale estimation is performed separately,
since items of different sub-scales are decoupled if we do not assume theta represents latent traits (i.e., we assume
a one-to-one correspondence between item sub-scales (e.g., Algebra, Geometry, etc.) and report card categories
(Algebra, Geometry, etc.)."""
import logging
import nirt.simulate.simulate_data
import nirt.solver
import numpy as np


if __name__ == "__main__":
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
    # For deterministic results.
    np.random.seed(0)
    # Number of persons.
    P = 100
    # Number of items.
    I = 20
    # A single sub-scale.
    C = 1
    # Using 2-PL model with fixed discrimination and no asymptote for all items.
    asym = 0  # 0.25
    discrimination = 1
    X, theta_exact, b, c = \
        nirt.simulate.simulate_data.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

    solver = nirt.solver.Solver(X, c, sample_size=20, num_iterations=5, num_sweeps=10)
    theta = solver.solve()
    np.set_printoptions(precision=2, linewidth=150, threshold=10000)
    error = np.linalg.norm(theta - theta_exact, axis=1)/np.linalg.norm(theta, axis=1)
    corr = np.corrcoef(np.squeeze(theta), np.squeeze(theta_exact))[0, 1]
    logger = logging.getLogger("__main__")
    #print(np.concatenate((theta, theta_exact, error[:,None]), axis=1))
    logger.info("Relative error |t-t'|/|t| = {:.2f}".format(np.linalg.norm(theta-theta_exact)/np.linalg.norm(theta)))
    logger.info("correlation {:.2f}".format(corr))
