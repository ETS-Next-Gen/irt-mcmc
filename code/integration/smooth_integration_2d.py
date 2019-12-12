"""A demo of numerically integrating a 2D IRT likelihood function by Gaussian quadratures.
This is the function from Sandip's paper, eq. (9):

p(t2>=t1|y1,y2) = \int \int_{t1}^{infty} L(t1;y1) L(t2;y2) p(t1,t2) dt2 dt1 /
                  \int \int L(t1;y1) L(t2;y2) p(t1,t2) dt2 dt1

where L is a Rasch model likelihood of observing the responses y given the latent ability theta.
Here y is a binary vector with k 1's and n-k non-zeros and L depends only on k and n, not on the
item ordering, as all items are assumed to have th same discrimination a and difficulty b.

p is a Gaussian with mean mu=0 and stddev sigma=1.
"""
import numpy as np
from matplotlib import use
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import fixed_quad


def irf(theta, a, b, k, n):
    """Item response function of n items, each of which has discrimination a, difficulty b.
    Returns the probability of k correct responses out of n."""
    return np.exp(k * a * (theta - b)) / (1 + np.exp(a*(theta - b))) ** n


def likelihood(theta, a, b, k, n):
    return irf(theta, a, b, k, n) * np.exp(-theta ** 2)


def t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=10, use_dynamic_programming=True):
    """Returns the integral p(t2>=t1|1,y2) with num_points quadrature points in each integral approximation."""
    # Generate nodes for t1 integration, which is Gauss-Hermite since it's f(t1) * exp(-t1^2) over [-inf,inf].
    node, weight = hermgauss(num_points)
    # For each t1, we calculate the integral over [t1, high] where high is slightly larger than the last t1 node.
    high = node[-1] + 1
    if use_dynamic_programming:
        # Dynamic programming: [t1, high] = [t1, t1'] + [t', high].
        integral_from_t1_to_inf = [0] * (num_points + 1)
        for i in range(num_points-1, -1, -1):
            s = fixed_quad(likelihood, node[i], node[i + 1] if i < num_points - 1 else high, args=(a, b, k2, n2), n=num_points)[0]
            integral_from_t1_to_inf[i] = integral_from_t1_to_inf[i + 1] + s
        integral_from_t1_to_inf = integral_from_t1_to_inf[:-1]
    else:
        # Calculate each integral [t1,high] separately from all other t1 values by a quadrature over this
        # interval. For the same number of points, this is less accurate since the interval is larger, so the
        # resolution is relatively lower than in the dynamic programming implementation.
        integral_from_t1_to_inf = [fixed_quad(likelihood, x, high, args=(a, b, k2, n2), n=num_points)[0] for x in node]

    values = [likelihood(theta, a, b, k1, n1) * f for theta, f in zip(node, integral_from_t1_to_inf)]
    return sum(values * weight)


def marginal_integral(a, b, k1, n1, k2, n2, num_points=10):
    """Returns the integral of the Likelihood function over [-inf,inf]^2 with num_points quadrature points in each
    integral approximation."""
    # Generate nodes for t1 integration, which is Gauss-Hermite since it's f(t1) * exp(-t1^2) over [-inf,inf].
    node, weight = hermgauss(num_points)
    t1_values = [likelihood(theta, a, b, k1, n1) for theta in node]
    t1_integral = sum(t1_values * weight)
    t2_values = [likelihood(theta, a, b, k2, n2) for theta in node]
    t2_integral = sum(t2_values * weight)
    return t1_integral * t2_integral


if __name__ == "__main__":
    a, b = 1, 0  # Item parameters.
    k1, n1 = 4, 10  # Representing the y1 data vector.
    k2, n2 = 6, 10  # Representing the y2 data vector.

    # Calculate the exact integrals (with many points).
    n = 30
    s_exact = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n)
    s_marginal_exact = marginal_integral(a, b, k1, n1, k2, n2, num_points=n)
    P_exact = s_exact / s_marginal_exact
    print("n %2d P %.5f" % (n, P_exact))

    # Print the error vs. #quadrature points.
    for n in range(5, 21):
        s = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n)
        s2 = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n, use_dynamic_programming=False)
        s_marginal = marginal_integral(a, b, k1, n1, k2, n2, num_points=n)
        P = s / s_marginal
        print("n %2d  DP %.5e (%.2e)  direct %.5e (%.2e)  marginal %.5e (%.2e)  P %.5f (%.2e)" %
              (n,
               s, np.abs(s - s_exact),
               s2, np.abs(s2 - s_exact),
               s_marginal, np.abs(s_marginal - s_marginal_exact),
               P, np.abs(P - P_exact)))
