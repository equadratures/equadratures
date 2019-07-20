from equadratures.parameter import Parameter
from equadratures.polyreg import Polyreg
from equadratures.polylsq import Polylsq
from equadratures.polyint import Polyint
from equadratures.polycs import Polycs
from equadratures.poly import Poly
from equadratures.stats import Statistics
from equadratures.basis import Basis
from equadratures.polynet import Polynet
from equadratures.nataf import Nataf
from equadratures.polynet import Polynet
from equadratures.utils import evalfunction, evalgradients, meshgrid
from equadratures.dr import *
from equadratures.optimization import Optimization
from equadratures.projectedpoly import Projectedpoly

import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

VERSION_NUMBER = 8.0
