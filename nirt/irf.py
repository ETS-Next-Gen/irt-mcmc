"""Non-parametric (binned) Item Response Function (IRF) building from thetas."""
import numpy as np

"""Range of item response function domain."""
M = 10


def item_response_function(X, t, c, n):
    tc = t[:, c]
    h = (2 * M)/n
    j = ((tc + M) / h).astype(int)
    # Anything off to the left is lumped into the left-most bin; similarly for the right boundary.
    j = np.minimum(np.maximum(j, 0), n-1)
    score = np.zeros((n,))
    count = np.zeros((n,), dtype=int)
    for p, jp in enumerate(j):
        score[jp] += X[p]
        count[jp] += 1
    return score, count
