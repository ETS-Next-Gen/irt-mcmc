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
    def __init__(self, grid: nirt.grid.Grid, x: np.array) -> None:
        # Calculate a simple histogram where each person contributes its full score value to its bin (generally: could
        # distribute a person's score into several neighboring bins).
        score = np.array([sum(x[b]) for b in grid.bin])
        count = np.array([len(x[b]) for b in grid.bin])

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
        self.interpolant = scipy.interpolate.interp1d(self.x, self.y, bounds_error=False, fill_value=(0, 1))

    @property
    def probability(self):
        """Returns the discrete IRF values at bin centers: P(X=1|bin_center[k]), k=1,...,num_bins."""
        return self.score / np.maximum(1e-15, self.count)

    def __repr__(self):
        return "count {} score {} P {}".format(self.count, self.score, self.probability)

    def plot(self, ax: plt.Axes, title: str = r"Item  Response Function", label: str = None, color: str = None,
             xlim=(-nirt.grid.M, nirt.grid.M)) -> None:
        """
        Draws the IRF interpolant and interpolation nodes and values.
        Args:
            ax: figure axis to draw in.
            title: str, optional, default: None. Plot title.
            label: str, optional. Default: None. Plot line label.

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
