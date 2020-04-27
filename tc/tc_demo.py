import logging
import bicluster.biclustering as bi
import cntree.cntree as cntree
import networkx as nx
import nirt.simulate.simulate_data as sim
import nirt.solver
import numpy as np
import scipy.sparse


def calculate_irf_proabilities(X, c):
    # Initial guess for thetas.
    t = nirt.likelihood.initial_guess(X, c)
    # Build IRFs.
    num_bins = 10
    method = "quantile"  # "uniform"
    grid = [nirt.grid.Grid(t[:, ci], num_bins, method=method) for ci in range(C)]
    irf = np.array([nirt.irf.ItemResponseFunction(grid[c[i]], X[:, i]) for i in range(I)])
    p = np.array([[irf[j].interpolant(t[i, c[j]]) for j in range(I)] for i in range(P)])
    return p, irf


def irf_error_at_nodes(irf, model):
    """Returns the scaled, weighted L2 norm of the error in the approximate IRF of item i at the nodes, averaged
    over all items i. Used to debug the IRT result."""
    mean_error = 0
    for i in range(I):
        # Calculate the
        f = irf[i]
        exact_irf = model[i](f.node)
        error = exact_irf - f.probability
        # Weight = bin count (so that all persons contribute the same weight to the norm: more
        # dense bins should count more)."""
        weight = f.count[f.has_data]
        e = (sum(weight * error ** 2) / sum(weight)) ** 0.5
        mean_error += e
    return mean_error / I


def aberration_biadjacency_matrix(X, p, p_max, s):
    """
    Builds a list of aberration graph edges. An edge weight represents the
    strength of deviation from the predicted probability of the person getting the item right, given that they did.
    This probability is exactly the IRF value.

    Args:
        X:
        p:
        p_max:
        s:

    Returns:
    w - sparse adjacency matrix between persons (rows) and items (columns).
    """
    row_ind, col_ind = np.where((X == 1) & (p <= p_max))
    w = scipy.sparse.csr_matrix((p[row_ind, col_ind] ** (-s), (row_ind, col_ind)), shape=X.shape)
    return w


def aberration_graph(w):
    """
    Builds the aberration graph. This is a bipartite graph between items and persons. Edge weight represents the
    strength of deviation from the predicted probability of the person getting the item right, given that they did.
    This probability is exactly the IRF value.

    Args:
        w:

    Returns:
    networkx bipartite graph object.
    """

    # Create aberration graph.
    P = w.shape[0]
    persons = np.arange(P, dtype=int)
    B = bipartite_graph(w, persons)
    # Filter graph to components that contain a significant number of items.
    large_components = [person for C in nx.connected_components(B)
                        if sum(1 for y in C if y >= P) > min_item_fraction * I
                        for person in C]
    B = B.subgraph(large_components)
    # Save this offset so we can easily distinguish between person and item nodes.
    B.P = P
    return B


def biadjacency_matrix(B):
    """Returns the bi-adjacency (person-item) matrix of the bipartite graph B."""
    n = np.sort(B.nodes())
    w = nx.bipartite.biadjacency_matrix(B, n[n < B.P], n[n >= B.P])
    return w


def bipartite_graph(w, persons):
    """Encode person nodes from 0..len(persons)-1 (where len(persons) <= P), and item nodes from P..P+I-1."""
    P = w.shape[0]
    B = nx.Graph()
    B.add_nodes_from(persons, bipartite=0)
    B.add_nodes_from(range(P, P + I), bipartite=1)
    row, col = w.nonzero()
    B.add_weighted_edges_from((p, P + i, wpi) for p, i, wpi in zip(row, col, w.data))
    return B


def draw_bipartite_graph(B):
    """Draws the graph."""
    top = nx.bipartite.sets(B)[0]
    pos = nx.bipartite_layout(B, top)
    nx.draw(B, pos)


if __name__ == "__main__":
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

    # Generate synthetic data, uni-dimensional latent trait.
    np.random.seed(0)
    P = 1000
    I = 20
    C = 1
    asym = 0
    discrimination = 1
    X, theta, b, c = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

    # Algorithm parameters.
    # Aberration accentuation parameter.
    s = 2.0
    # Only probabilities up to this value are considered for aberration graph links.
    p_max = 0.3
    # A collusion group must correspond to at least this fraction of leaked items.
    min_item_fraction = 0.2

    # Use IRT to estimate the probability that a person gets an item right.
    p, irf = calculate_irf_proabilities(X, c)
    # Check IRF accuracy (since this is synthetic data we can calculate this here).
    model = [lambda x, b=b[i]: sim.three_pl_model(x, discrimination, b, asym) for i in range(I)]
    error = irf_error_at_nodes(irf, model)
    print("Mean IRF accuracy {:.2f}".format(error))

    # Build the aberration graph.
    w = aberration_biadjacency_matrix(X, p, p_max, s)
    B = aberration_graph(w)
    w = biadjacency_matrix(B)

    # Filter persons to those that have edges in the graph (candidate cheaters).
    y, z, clustering = bi.bicluster(w, max_cluster_radius=0.01, max_cluster_size=0)
    for level in cntree.get_clustering_levels(clustering):
        print(level)
