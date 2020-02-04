# Simulate MCMC model data: pick the true parameter variables from the model's prior distribution.
import cluster.cntree.cntree
import nirt.irf
import logging
import numpy as np
import numpy.matlib
from scipy.stats import invgamma


def set_simple_logging(level):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")


def generate_simulated_data(P, I, C, asym=0.25, discrimination=1):
    """
    Generates simulated IRT data. Generates binary item responses. Assumes theta[:,c], c=1,...,C
    follows a N(0, InvGamma(1,1)) distribution. Item difficulties are uniformly spaced. Item discrimination is
    uniformly random in [0.5,1.5]. Item classification is defined so that there are roughly equal number of items
    per class.

    Args:
        P: int, number of persons.
        I: int, number of items.
        C: int, number of item classes = number of latest abililty dimensions.
        asym: float, probability of guessing = asymptotic IRF value at theta -> -\infty.

    Returns:
        x: array<float>, shape=(P, I) binary item response matrix.
        theta: array<float>, shape=(P, C) latent person ability matrix.
        c: (I,): array<int> item classification (0 <= c[i] < C).
    """
    # Generate latent ability distribution variances.
    alpha_theta, beta_theta = 1, 1
    rv = invgamma(a=alpha_theta, scale=beta_theta)
    v = rv.rvs(C)

    # Generate normally distributed student latent abilities. theta_c ~ N(0, invgamma(a_c,b_c))
    cov = np.diag(v)
    theta = np.random.multivariate_normal(np.zeros((C,)), cov, P)

    # Generate item difficulty parameters.
    # Discrimination is uniform[0.5, 1.5].
    if discrimination is None:
        a = np.random.random(size=(I,)) + 0.5
    else:
        a = np.ones((I,)) * discrimination
    # Difficulty is equally spaced from -3 to 3.
    b = np.linspace(-3, 3, num=I)
    # Item i measures sub-scale c[i]. Select about the same number of items per subscale,
    # then randomly permute the item order.
    c = np.random.permutation(np.matlib.repmat(np.arange(C, dtype=int), int(np.ceil(I / C)), 1).ravel()[:I])

    # Generate item responses (the observed data) (3PL model).
    t = np.exp(a*(theta[:, c] - b))
    p_correct = asym + (1-asym) * (t/(1+t))
    x = np.random.binomial(1, p=p_correct)
    return x, theta, b, c


def cluster_persons(data, initial_num_clusters):
    """
    Hierarchically clusters persons based on their item responses. The persons are first clustered into
    'initial_num_clusters' groups, which are subsequently broken into smaller clusters, etc. The last level has
    clusters of size <= 2, so that the next clustering level (not included in the returned object) consists of the
    original, individual persons.

    We use Euclidean metric + K-means at every level, regardless of the type of 'data' (binary/continuous scores).

    Args:
        data: np.ndarray.array item response data, shape: num_persons x num_items.
        initial_num_clusters: size of first clustering level.

    Returns:
        cluster.cntree.cntree.Level clsutering object.
    """
    x = data.astype(float)
    tree = cluster.cntree.cntree.CNTree(
        max_cluster_radius=0, max_cluster_size=2, debug=1,
        branch_factor=2,
        initial_children="principal_direction", initial_num_local_iters=0)
    return tree.cluster(x)


def three_pl_model(theta, a, b, asym):
    t = np.exp(a * (theta - b))
    p_correct = asym + (1 - asym) * (t / (1 + t))
    return p_correct


def plot_irf(ax, i, item_params, score, count, n):
    a, b, c, asym = item_params
    M = nirt.irf.M
    h = (2 * M) / n
    theta_range = np.linspace(-M + h / 2, M - h / 2, n)
    has_data = count > 0
    irf = score[has_data] / count[has_data]
    original_irf = lambda t: three_pl_model(t, a, b, asym)
    t_continuous = np.linspace(-M, M, 100)
    ax.scatter(theta_range, original_irf(theta_range), color="blue", s=30, label="Original")
    ax.scatter(theta_range[has_data], irf, color="red", s=30, label="Estimated")
    ax.plot(t_continuous, original_irf(t_continuous), color="blue")
    ax.set_title("IRF i={}".format(i))
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
