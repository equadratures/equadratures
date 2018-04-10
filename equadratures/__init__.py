'Initialization file'
from .parameter import Parameter
from .polyreg import Polyreg
from .polycs import Polycs
from .polyint import Polyint
from .polylsq import Polylsq
from .stats import Statistics
from .basis import Basis
import numpy as np
import equadratures.qr as qr
from .plotting import *
from .utils import evalfunction, evalgradients, meshgrid
from .dr import *
from .distributions import *
import matplotlib
params = {'legend.fontsize': 18,
          'axes.labelsize': 18,
          'axes.titlesize': 18,
          'xtick.labelsize' :12,
          'ytick.labelsize': 12,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'sans',
          'font.variant':'small-caps',
          'grid.linestyle': ':',
          'grid.linewidth': 0.5,
         }
matplotlib.rcParams.update(params)
