"""An adaptive grid of bins of theta values in one dimension. Instead of a uniform grid, we now use percentiles, to
maximize and balance the sample size per bin."""
import nirt.grid
import numpy as np
from typing import Tuple

"""Limited for fixed uniform domain for theta: [-M,M]."""
M = 8


