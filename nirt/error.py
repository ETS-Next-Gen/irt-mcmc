"""IRF and theta error metrics, which can be calculated for simulated data."""
import numpy as np


def error_norm(model_irf, irf):
    """
    Returns the scaled, weighted L2 norm of the error in the approximate IRF at the nodes.
    Weight = bin count (so that all persons contribute the same weight to the norm: more
    dense bins should count more).

    Args:
        model_irf: exact parametric IRF form.
        irf: numerical IRF to check.

    Returns: ||error_irf - irf||.
    """
    exact_irf = np.array([model_irf(t) for t in irf.node])
    error = exact_irf - irf.probability
    weight = irf.count
    return (sum(weight * error ** 2) / sum(weight)) ** 0.5


def error_norm_by_item(model_irf, irf):
    """
    Returns the scaled, weighted L2 norm of the error in the approximate IRF at the nodes vs. a model, for each
    element of the array 'irf'.

    Args:
        model_irf: array of exact parametric IRFs.
        irf: array of numerical IRFs to check.

    Returns: array of errors ||error_irf[i] - irf[i]||.
    """
    return np.array([error_norm(f, g) for f, g in zip(model_irf, irf)])