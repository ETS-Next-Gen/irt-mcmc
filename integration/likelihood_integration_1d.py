"""A demo of numerically integrating a smooth function (e.g., an IRT likelihood function) by Gaussian
quadratures."""
import numpy as np
from scipy.integrate import fixed_quad, quadrature


def theta_distribution(theta, mu, sigma):
    """Latent ability distribution."""
    return np.exp(-(theta-mu)**2 / sigma**2)


def irf(theta, a, b):
    """Item response function."""
    t = np.exp(a*(theta - b))
    return t / (1 + t)


def likelihood(theta, a, b, mu, sigma):
    """Likelihood of a correct answer on a single item."""
    return irf(theta, a, b) * theta_distribution(theta, mu, sigma)


def test_integration_error(low, high):
    """Prints integration error of Gaussian quadratures for integrating the likelihood over [lo,high]."""
    a, b = 1, 1
    mu, sigma = 0, 1
    print("Integrating likelihood function with item parameters "
          "a = {}, b = {}, theta = Normal({}, {}) over [{},{}]".format(a, b, mu, sigma, low, high))
    s, error = quadrature(likelihood, low, high, args=(a, b, mu, sigma), tol=1e-08, rtol=1e-08, maxiter=50)
    print("Fixed-tolerance quadrature:      result = %.8f  error = %.2e" % (s, error))
    for n in range(5, 21):
        sn, _ = fixed_quad(likelihood, low, high, args=(a, b, mu, sigma), n=n)
        print("Fixed-point quadrature: n = %2d   result = %.8f  error = %.2e" % (n, sn, np.abs(s - sn)))


if __name__ == "__main__":
    for lo in (-3, -2, -1, 0):
        test_integration_error(lo, 6)
