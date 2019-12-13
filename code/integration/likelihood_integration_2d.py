"""A demo of numerically integrating a 2D IRT likelihood function by Gaussian quadratures.
This is the function from Sandip's paper, eq. (9):

p(t2>=t1|y1,y2) = \int \int_{t1}^{infty} L(t1;y1) L(t2;y2) p(t1,t2) dt2 dt1 /
                  \int \int L(t1;y1) L(t2;y2) p(t1,t2) dt2 dt1

where L is a Rasch model likelihood of observing the responses y given the latent ability theta.
Here y is a binary vector with k 1's and n-k non-zeros and L depends only on k and n, not on the
item ordering, as all items are assumed to have th same discrimination a and difficulty b.

p is a Gaussian with mean mu=0 and stddev sigma=1.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import fixed_quad


def irf(theta: float, a: float, b: int, k: int, n: int) -> float:
    """
    Returns the item response function of n dichotomous items of which k were correctly answered.
    All items are assumed to have the same parameters.
    :param theta: student latent ability.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k: #correct responses.
    :param n: #items.
    :return: probability of getting k correct responses out of n given the ability theta.
    """
    return np.exp(k * a * (theta - b)) / (1 + np.exp(a*(theta - b))) ** n


def likelihood(theta: float, a: float, b: float, k: int, n: int) -> float:
    """
    Returns the likelihood function L(theta;X) proportional to P(X|theta) P(theta).
    :param theta: student latent ability.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k: #correct responses.
    :param n: #items.
    :return: likelihood of getting k correct responses out of n given the ability theta given a Rasch
    item response function (irf()) and a normal latent ability prior distribution.
    """
    return irf(theta, a, b, k, n) * np.exp(-theta ** 2)


def t2_gt_t1_integral(a: float, b: float, k1: int, n1: int, k2: int, n2: int,
                      num_points=10, use_dynamic_programming=True) -> float:
    """
    Returns the likelihood integral int int L(t1,t2|y1,y2) over the domain t2>=t1 with 'num_points' quadrature points
    in each integral approximation. y1 represents a student that answered k1 out of n1 items correctly.
    y2 represents a student that answered k2 out of n2 items correctly. t1 and t2 are the students' abilities,
    respectively. All items are assumed to have the same parameters. A normal latent ability prior distribution is
    assumed.
    :param theta: student latent ability.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k1: #correct responses of student 1.
    :param n1: #items of student 1.
    :param k2: #correct responses of student 2.
    :param n2: #items of student 2.
    :param num_points: #quadrature points to use in each integral approximation (t1 and t2 integrals).
    :param use_dynamic_programming: iff True, uses dynamic programming to calculate int_t1^{inf} L(...) dt2 for
    different t1 values.
    :return: the likelihood integral int int L(t1,t2|y1,y2) over the domain t2>=t1.
    """
    # Generate nodes for t1 integration, which is Gauss-Hermite since it's f(t1) * exp(-t1^2) over [-inf,inf].
    node, weight = hermgauss(num_points)
    # For each t1, we calculate the integral over [t1, high] where high is slightly larger than the last t1 node.
    high = node[-1] + 1
    if use_dynamic_programming:
        # Dynamic programming: [t1, high] = [t1, t1'] + [t', high].
        integral_from_t1_to_inf = [0] * (num_points + 1)
        for i in range(num_points-1, -1, -1):
            s = fixed_quad(likelihood, node[i], node[i + 1] if i < num_points - 1 else high,
                           args=(a, b, k2, n2), n=num_points)[0]
            integral_from_t1_to_inf[i] = integral_from_t1_to_inf[i + 1] + s
        integral_from_t1_to_inf = integral_from_t1_to_inf[:-1]
    else:
        # Calculate each integral [t1,high] separately from all other t1 values by a quadrature over this
        # interval. For the same number of points, this is less accurate since the interval is larger, so the
        # resolution is relatively lower than in the dynamic programming implementation.
        integral_from_t1_to_inf = [fixed_quad(likelihood, x, high, args=(a, b, k2, n2), n=num_points)[0] for x in node]

    values = [irf(theta, a, b, k1, n1) * f for theta, f in zip(node, integral_from_t1_to_inf)]
    return sum(values * weight)


def marginal_integral(a: float, b: float, k1: int, n1: int, k2: int, n2: int, num_points=10) -> float:
    """
    Returns the marginal likelihood integral int int L(t1,t2|y1,y2) with 'num_points' quadrature points
    in each integral approximation. y1 represents a student that answered k1 out of n1 items correctly.
    y2 represents a student that answered k2 out of n2 items correctly. t1 and t2 are the students' abilities,
    respectively. All items are assumed to have the same parameters. A normal latent ability prior distribution is
    assumed.
    :param theta: student latent ability.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k1: #correct responses of student 1.
    :param n1: #items of student 1.
    :param k2: #correct responses of student 2.
    :param n2: #items of student 2.
    :param num_points: #quadrature points to use in each integral approximation (t1 and t2 integrals).
    :param use_dynamic_programming: iff True, uses dynamic programming to calculate int_t1^{inf} L(...) dt2 for
    different t1 values.
    :return: the likelihood integral int int L(t1,t2|y1,y2) over all t1, t2.
    """
    # Generate nodes for t1 integration, which is Gauss-Hermite since it's f(t1) * exp(-t1^2) over [-inf,inf].
    node, weight = hermgauss(num_points)
    t1_values = [irf(theta, a, b, k1, n1) for theta in node]
    t1_integral = sum(t1_values * weight)
    t2_values = [irf(theta, a, b, k2, n2) for theta in node]
    t2_integral = sum(t2_values * weight)
    return t1_integral * t2_integral


def probability_t2_gt_t1(a: float, b: float, k1: int, n1: int, k2: int, n2: int, num_points=10) -> float:
    """
    Returns the probability p(t2>=t1|y1,y2). y1 represents a student that answered k1 out of n1 items correctly.
    y2 represents a student that answered k2 out of n2 items correctly. t1 and t2 are the students' abilities,
    respectively. All items are assumed to have the same parameters. A normal latent ability prior distribution is
    assumed.
    :param theta: student latent ability.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k1: #correct responses of student 1.
    :param n1: #items of student 1.
    :param k2: #correct responses of student 2.
    :param n2: #items of student 2.
    :param num_points: #quadrature points to use in each integral approximation (t1 and t2 integrals).
    :return: the likelihood integral int int L(t1,t2|y1,y2) over all t1, t2.
    """
    s = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=num_points)
    marginal = marginal_integral(a, b, k1, n1, k2, n2, num_points=num_points)
    return s / marginal


def test_quadrature_accuracy_decreases_with_n(a: float, b: float, k1: int, n1: int, k2: int, n2: int):
    """
    Tests the integrals' accuracy vs. #quadrature points.
    :param a: item discrimination.
    :param b: item difficulty.
    :param k1: #correct responses of student 1.
    :param n1: #items of student 1.
    :param k2: #correct responses of student 2.
    """
    # Calculate the exact integrals (with many points).
    n = 30
    s_exact = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n)
    s_marginal_exact = marginal_integral(a, b, k1, n1, k2, n2, num_points=n)
    P_exact = s_exact / s_marginal_exact
    print("k1 %2d n1 %2d k2 %2d n2 %2d" % (k1, n1, k2, n2))
    print("n %2d  numerator %.5e  marginal %.5e  P %.5f" %
          (n, s_exact, s_marginal_exact, P_exact))

    # Print the error vs. #quadrature points.
    for n in range(5, 21):
        s = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n, use_dynamic_programming=True)
        s2 = t2_gt_t1_integral(a, b, k1, n1, k2, n2, num_points=n, use_dynamic_programming=False)
        s_marginal = marginal_integral(a, b, k1, n1, k2, n2, num_points=n)
        P = s / s_marginal
        print("n %2d  DP %.5e (%.2e)  direct %.5e (%.2e)  marginal %.5e (%.2e)  P %.5f (%.2e)" %
              (n,
               s, np.abs(s - s_exact),
               s2, np.abs(s2 - s_exact),
               s_marginal, np.abs(s_marginal - s_marginal_exact),
               P, np.abs(P - P_exact)))


def probability_t2_gt_t1_table(a: float, b: float, n: int) -> np.ndarray:
    """
    Returns a table of P(t2>=t1) values vs. k1, k2 for n items.
    :param a: item discrimination.
    :param b: item difficulty.
    :param n1: #items.
    :return: array of size (n+1)x(n+1): P[k1,k2] = P(t2>=t1|y1=k1,y2=k2).
    """
    return np.array([[probability_t2_gt_t1(a, b, k1, n, k2, n, num_points=30) for k2 in range(n + 1)]
                     for k1 in range(n + 1)])


def test_probability_t2_gt_t1_is_sane():
    """Creates a table of P(t2>=t1) values vs. k1, k2 for 10 items. Verifies that 0 <= P <= 1, P(k,k) = 0.5,
    P(k2,k1) = 1 - P(k1,k2). P should be increasing with k1 for fixed k2.
    """
    P = probability_t2_gt_t1_table(1, 0, 10)
    assert np.all(P >= 0)
    assert np.all(P <= 1)
    assert np.allclose(P, 1 - P.transpose())
    assert np.all(np.diff(p, axis=1) >= 0)


if __name__ == "__main__":
    a, b = 1, 0  # Item parameters.

    # The example requested by Sandip.
    test_quadrature_accuracy_decreases_with_n(a, b, 4, 10, 6, 10)
    test_probability_t2_gt_t1_is_sane()

    # Print and plot a table of P(t2>=t1) values vs. k1, k2 for 10 items.
    # P must satisfy 0 <= P <= 1, P(k,k) = 0.5, P(k2,k1) = 1 - P(k1,k2).
    n = 10
    np.set_printoptions(linewidth=200, precision=5, suppress=True)
    P = probability_t2_gt_t1_table(a, b, n)
    print(P)
    plt.figure(1)
    plt.clf()
    plt.plot(range(n + 1), P.transpose())
    plt.grid(True)
    plt.legend(["$k_1 = %d$" % k1 for k1 in range(n + 1)], prop={'size': 5})
    plt.xlabel("$k_2$ (#correct items of student 2)")
    plt.ylabel("$P[t_2 \geq t_1|y_1=k_1/n, y_2=k_2/n]$")
    plt.title("Probability of $t_2 \geq t_1$ for $n = %d$ items" % n)
    plt.savefig("prob_t2_gt_g1.png")
