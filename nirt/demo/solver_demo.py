import logging
import nirt.simulate.simulate_data
import nirt.solver
import numpy as np


if __name__ == "__main__":
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

    np.random.seed(0)

    # Number of persons.
    P = 100
    # Number of items.
    I = 20
    # Number of latent ability dimensions (sub-scales).
    C = 5
    # Using 2-PL model with fixed discrimination and no asymptote for all items.
    asym = 0  # 0.25
    discrimination = 1
    X, theta_exact, b, c = \
        nirt.simulate.simulate_data.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

    solver = nirt.solver.Solver(X, c, num_iterations=5, num_sweeps=5)
    theta = solver.solve()
    np.set_printoptions(precision=2, linewidth=150, threshold=10000)
    error = np.linalg.norm(theta - theta_exact, axis=1)/np.linalg.norm(theta, axis=1)
    corr = np.array([np.corrcoef(theta[p], theta_exact[p])[0, 1] for p in range(P)])
    print(np.concatenate((theta, theta_exact, error[:,None], corr[:,None]), axis=1))
    print(np.linalg.norm(theta-theta_exact)/np.linalg.norm(theta))

    p = -1
    print(theta[p])
    print(theta_exact[p])
    print(np.corrcoef(theta[p], theta_exact[p]))
