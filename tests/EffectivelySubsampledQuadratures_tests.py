#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
from effective_quadratures.Effective_Subsampling import EffectiveSubsampling
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import numpy.ma as maogle
import os
"""

    Testing Script for Effective Quadrature Suite of Tools

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
# Simple analytical function
def fun(x):
    return np.exp(x[0] + x[1])

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 6
    derivative_flag = 0 # derivative flag
    min_value, max_value = -1, 1

    # Decide on the polynomial basis. We recommend total order or hyperbolic cross
    # basis terms. First we create an index set object
    hyperbolic_basis = IndexSet("hyperbolic cross", [order, order])

    # The "UQ" parameters
    uq_parameters = []
    uniform_parameter = PolynomialParam("Uniform", min_value, max_value, [], [] , derivative_flag, order)
    uq_parameters.append(uniform_parameter)
    uq_parameters.append(uniform_parameter)

    # Define the EffectiveSubsampling object and get "A"
    effectiveQuads = EffectiveSubsampling(uq_parameters, hyperbolic_basis, derivative_flag)
    A = EffectiveSubsampling.getA(effectiveQuads)
    print A

main()
