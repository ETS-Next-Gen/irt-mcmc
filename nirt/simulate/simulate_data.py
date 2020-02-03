# Simulate MCMC model data: pick the true parameter variables from the model's prior distribution.
import cluster.cntree.cntree
import logging
import numpy as np
import numpy.matlib
from scipy.stats import invgamma


def set_simple_logging(level):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")


def generate_simulated_data(P, I, C, asym):

    # Generate latent ability distribution variances.
    alpha_theta, beta_theta = 1, 1
    rv = invgamma(a=alpha_theta, scale=beta_theta)
    v = rv.rvs(C)
    print(v)

    # Generate normally distributed student latent abilities. theta_c ~ N(0, invgamma(a_c,b_c))
    cov = np.diag(v)
    theta = np.random.multivariate_normal(np.zeros((C,)), cov, P)

    # Generate item difficulty parameters.
    # Discrimination is uniform[0.5, 1.5].
    a = np.random.random(size=(I,)) + 0.5
    # Difficulty is equally spaced from -3 to 3.
    b = np.linspace(-3, 3, num=I)
    # Item i measures sub-scale c[i]. Select about the same number of items per subscale,
    # then randomly permute the item order.
    c = np.random.permutation(np.matlib.repmat(np.arange(C, dtype=int), int(np.ceil(I/C)), 1).ravel()[:I])

    # Generate item responses (the observed data).
    t = np.exp(-a*(theta[:,c] - b))
    p_correct = asym + (1-asym)*(t/(1+t))
    X = np.random.binomial(1, p=p_correct)
    return X, theta


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


if __name__ == "__main__":
    set_simple_logging(logging.DEBUG)

    # Number of persons.
    P = 1000
    # Number of items.
    I = 40
    # Number of latent ability dimensions (sub-scales).
    C = 5
    # Fixed item asymptote (pi) as theta -> -\infty = probability of guessing.
    asym = 0.25

    X, theta = generate_simulated_data(P, I, C, asym)
    print(X)
    clustering = cluster_persons(X, C)
    print(clustering)