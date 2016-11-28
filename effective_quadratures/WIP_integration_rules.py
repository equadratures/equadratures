#!/usr/bin/env python
"""Set of tools for 1D integration rules -- File name to be varied later!"""
from parameter import Parameter
from polynomial import Polynomial
from indexset import IndexSet
from effectivequads import EffectiveSubsampling
import numpy as np
#****************************************************************************
# Functions to code:
#    
# 1. Faster Gauss quadrature rules (see work by Alex Townsend & Nick Hale)
# 2. Sparse grid quadrature rules with different growth rules
# 3. Spherical quadrature rules
# 4. Padua quadrature rules -- add 2D!
#****************************************************************************