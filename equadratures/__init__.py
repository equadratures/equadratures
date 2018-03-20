'Initialization file'
from .parameter import Parameter
from .polyreg import Polyreg
from .polycs import Polycs
from .stats import Statistics
from .basis import Basis
import numpy as np
import equadratures.qr as qr 
from .plotting import *
from .utils import evalfunction, evalgradients, meshgrid
from .dimension_reduction import *
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
#matplotlib.rcParams.update({'text.usetex':True, 'text.latex.preamble':[r'\usepackage{amsmath, newtxmath}']})