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
params = {'legend.fontsize': 11,
          'font.size' : 10.0,
          'font.family': 'serif',
          'font.stretch': 'semi-condensed',
          'axes.labelsize': 11,
          'axes.titlesize': 11,
          'axes.axisbelow': True,
          'xtick.labelsize' :11,
          'ytick.labelsize': 11,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'sans',
          'font.variant':'small-caps',
          'grid.linestyle': '-',
          'grid.color': 'white',
          'grid.linewidth': 2.0,
          'axes.spines.right':False,
          'axes.spines.top': False,
          'axes.grid': True,
          'axes.facecolor':'whitesmoke',
          'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.frameon' : False,
          'image.cmap' : 'gist_earth'
          #'grid.linewidth': 0.5,
         }
matplotlib.rcParams.update(params)
