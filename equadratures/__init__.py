from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.stats import Statistics
from equadratures.basis import Basis
from equadratures.polynet import Polynet
from equadratures.utils import evalfunction, evalgradients, meshgrid
from equadratures.dr import *
from equadratures.optimization import Optimization
from equadratures.mesh import Mesh
from equadratures.projectedpoly import Projectedpoly
from equadratures.induced_distributions import InducedSampling

import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

VERSION_NUMBER = 8.0
