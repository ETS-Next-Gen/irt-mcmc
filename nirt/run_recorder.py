"""Records IRFs and theta estimates during a solver run."""
import collections
import logging
import nirt.irf
import nirt.grid
import nirt.likelihood
import nirt.mcmc
import nirt.theta_improvement
import numpy as np


class RunRecorder:
    """Records IRFs and theta estimates during a solver run."""
    def __init__(self):
        self.theta = collections.OrderedDict()
        self.irf = collections.OrderedDict()

    def add_theta(self, num_bins, theta):
        pass

    def add_irf(self, num_bins, irf):
        pass
