# Simulate MCMC model data: pick the true parameter variables from the model's prior distribution.
import nirt.irf
import logging
import numpy as np
import numpy.matlib
from scipy.stats import invgamma


# def cluster_persons(data, initial_num_clusters):
#     """
#     Hierarchically clusters persons based on their item responses. The persons are first clustered into
#     'initial_num_clusters' groups, which are subsequently broken into smaller clusters, etc. The last level has
#     clusters of size <= 2, so that the next clustering level (not included in the returned object) consists of the
#     original, individual persons.
#
#     We use Euclidean metric + K-means at every level, regardless of the type of 'data' (binary/continuous scores).
#
#     Args:
#         data: np.ndarray.array item response data, shape: num_persons x num_items.
#         initial_num_clusters: size of first clustering level.
#
#     Returns:
#         cluster.cntree.cntree.Level clsutering object.
#     """
#     x = data.astype(float)
#     tree = cluster.cntree.cntree.CNTree(
#         max_cluster_radius=0, max_cluster_size=2, debug=1,
#         branch_factor=2,
#         initial_children="principal_direction", initial_num_local_iters=0)
#     return tree.cluster(x)


def three_pl_model(theta, a, b, asym):
    t = np.exp(a * (theta - b))
    p_correct = asym + (1 - asym) * (t / (1 + t))
    return p_correct


def plot_model_irf(ax, grid, model_irf, color="black", label=None, xlim=None):
    if xlim is None:
        xlim = (grid.range[0] - 1, grid.range[1] + 1)
    ax.scatter(grid.center, model_irf(grid.center), color=color, s=30, label=label)
    t_continuous = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(t_continuous, model_irf(t_continuous), color=color)
    ax.set_ylim([-0.1, 1.1])


def plot_discrete_irf(ax, irf, n, color="black", label=None):
    score, count = irf.score, irf.count
    theta_range = nirt.irf.bin_centers(n)
    has_data = count > 0
    p = score[has_data] / count[has_data]
    ax.scatter(theta_range[has_data], p, color=color, s=30, label=label)
    ax.set_ylim([-0.1, 1.1])
