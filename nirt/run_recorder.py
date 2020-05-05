"""Records IRFs and theta estimates during a solver run."""
import collections


class RunRecorder:
    """Records IRFs and theta estimates during a solver run."""
    def __init__(self):
        self.theta = collections.OrderedDict()
        self.irf = collections.OrderedDict()

    def add_theta(self, num_bins, theta):
        self.theta.setdefault(num_bins, []).append(theta)

    def add_irf(self, num_bins, irf):
        self.irf.setdefault(num_bins, []).append(irf)
