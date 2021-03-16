"""Builds non-parametric (binned) Item Response Function (IRF) from thetas."""
import nirt.grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


class ItemResponseFunction:
    """An Item Response Function (IRF): a numerical histogram of success on the item for a population of person latent
    abilities. The IRF is a linear interpolant of histogram values at theta bin centers.

    Attributes:
        grid: nirt.grid.Grid, holds bins of person thetas (x-axis of discrete IRF).
        score: array<int>, shape=(num_bins,), total score of all persons in a bin, for each bin.
        count: array<int>, shape=(num_bins,), total count of all persons in a bin, for each bin.
        interpolant: function, IRF interpolant.
        x: array<float> interpolation nodes (bin center + extension points).
        y: array<float> interpolation values.
    """
    def __init__(self, grid: nirt.grid.Grid, x: np.array, num_smoothing_sweeps: int = 0,
                 histogram: str = "simple") -> None:
        # Calculate a simple histogram where each person contributes its full score value to its bin (generally: could
        # distribute a person's score into several neighboring bins).
        if histogram == "simple":
            score, count = simple_histogram(grid, x)
        elif histogram == "distributed":
            score, count = linearly_distributed_histogram(grid, x)
        else:
            raise ValueError("Unsupported histogram type '{}'".format(histogram))

        # Filter empty bins. (In an adaptive quantile grid no bins should be empty, but generally this can happen.)
        has_data = count > 0
        self.score = score[has_data]
        self.count = count[has_data]
        self.grid = grid

        # Create a linear interpolant from score values at bin centers. Extend IRF as 0 to the left and 1 to the right.
        node = grid.center[has_data]
        self.node = node
        self.has_data = has_data
        self.x = np.concatenate(([2 * node[0] - node[1]], node, [2 * node[-1] - node[-2]]))
        self.y = np.concatenate(([0], self.probability, [1]))
        for _ in range(num_smoothing_sweeps):
            relax(self.y)
        self.interpolant = scipy.interpolate.interp1d(self.x, self.y, bounds_error=False, fill_value=(0, 1))

    @property
    def probability(self):
        """Returns the discrete IRF values at bin centers: P(X=1|bin_center[k]), k=1,...,num_bins."""
        return self.score / np.maximum(1e-15, self.count)

    def __repr__(self):
        return "count {} score {} P {}".format(self.count, self.score, self.probability)

    def plot(self, ax: plt.Axes, title: str = r"Item Response Function", label: str = None, color: str = None,
             xlim=(-nirt.grid.M, nirt.grid.M)) -> None:
        """
        Draws the IRF interpolant and interpolation nodes and values.
        Args:
            ax: figure axis to draw in.
            title: str, optional, default: None. Plot title.
            label: str, optional. Default: None. Plot line label.
            color: str, optional. Default: None.

        Returns: None.
        """
        # Draw the interpolation nodes (bin centers + extension nodes).
        if xlim is None:
            xlim = (self.x[0], self.x[-1])
        t = np.linspace(xlim[0], xlim[1], 10 * len(self.x) + 1)
        ax.plot(t, self.interpolant(t), label=label, color=color)
        ax.plot(self.x, self.y, color=color, marker="o")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$P(X=1|\theta)$")
        ax.set_title(title)


def relax(y):
    for i in range(1, len(y) - 1):
        y[i] = 0.5 * (y[i - 1] + y[i + 1])


def simple_histogram(grid: nirt.grid.Grid, x: np.array):
    score = np.array([sum(x[b]) for b in grid.bin])
    count = np.array([len(x[b]) for b in grid.bin])
    return score, count


def linearly_distributed_histogram(grid: nirt.grid.Grid, x: np.array):
    theta = grid._theta
    score = np.zeros((len(grid.bin),))
    count = np.zeros((len(grid.bin),))
    for b, grid_bin in enumerate(grid.bin):
        theta_bin = theta[grid_bin]
        score_bin = x[grid_bin]

        p = theta_bin < grid.center[b]
        theta_p = theta_bin[p]
        if b == 0:
            score[b] += sum(score_bin[p])
            count[b] += len(theta_p)
        else:
            distance_center_left = grid.center[b] - theta_p
            distance_center_right = theta_p - grid.center[b - 1]
            score[b - 1] += sum((distance_center_left * score_bin[p]) / (distance_center_left + distance_center_right))
            count[b - 1] += sum(distance_center_left / (distance_center_left + distance_center_right))
            score[b] += sum((distance_center_right * score_bin[p]) / (distance_center_left + distance_center_right))
            count[b] += sum(distance_center_right / (distance_center_left + distance_center_right))

        p = theta_bin >= grid.center[b]
        theta_p = theta_bin[p]
        if b == len(grid.bin) - 1:
            score[b] += sum(score_bin[p])
            count[b] += len(theta_p)
        else:
            distance_center_left = theta_p - grid.center[b]
            distance_center_right = grid.center[b + 1] - theta_p
            score[b] += sum((distance_center_left * score_bin[p]) / (distance_center_left + distance_center_right))
            count[b] += sum(distance_center_left / (distance_center_left + distance_center_right))
            score[b + 1] += sum((distance_center_right * score_bin[p]) / (distance_center_left + distance_center_right))
            count[b + 1] += sum(distance_center_right / (distance_center_left + distance_center_right))
    return score, count


def sorted_chunks(x, bin_size):
    perm = np.argsort(x.mean(axis=1))
    return np.array([perm[i: i + bin_size] for i in range(0, bin_size * (x.shape[0] // bin_size), bin_size)])