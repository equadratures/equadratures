import distributions, parameter, poly, polycs, polyint, polyreg, polylsq, basis, stats, nataf
from parameter import Parameter 
from polyreg import Polyreg 
from polylsq import Polylsq
from polyint import Polyint 
from polycs import Polycs
from poly import Poly 
from stats import Statistics
from basis import Basis 
from nataf import Nataf
import numpy as np
from utils import evalfunction, evalgradients, meshgrid
from dr import *
import matplotlib
params = {'legend.fontsize': 13,
          'font.size' : 13.0,
          'font.family': 'serif',
          'font.stretch': 'semi-condensed',
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'axes.axisbelow': True,
          'xtick.labelsize' :13,
          'ytick.labelsize': 13,
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
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='y', colors='black', width=2)
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='x', colors='black', width=2)
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])