"""A demo of numerically integrating a smooth function (e.g., an IRT likelihood function) by Gaussian
quadratures."""
import numpy as np
from scipy.integrate import fixed_quad, quadrature

def theta_distribution(theta, mu, sigma):
    return np.exp(-(x-mu)**2 / sigma**2)

def irf(theta, a, b):
    t = np.exp(a*(theta - b))
    return t / (1 + t)

def likelihood(theta, a, b, mu, sigma):
    return irf(theta, a, b) * irf(theta, mu, sigma)

if __name__ == "__main__":
    a, b = 1, 1
    mu, sigma = 0, 1

    print("Integrating likelihood function with item parameters a = {}, b = {}, theta = Normal({}, {})".format(a, b, mu, sigma))

    s, error = quadrature(likelihood, -1, 6, args=(a, b, mu, sigma), tol=1e-08, rtol=1e-08, maxiter=50)
    print("Fixed-tolerance quadrature:      result = %.8f  error = %.2e" % (s, error))

    for n in range(5, 11):
        sn, _ = fixed_quad(likelihood, -1, 6, args=(a, b, mu, sigma), n=n)
        print("Fixed-point quadrature: n = %2d   result = %.8f  error = %.2e" % (n, sn, np.abs(s - sn)))