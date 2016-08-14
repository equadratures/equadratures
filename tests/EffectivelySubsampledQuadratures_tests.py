#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
import effective_quadratures.ComputeStats as stats
from effective_quadratures.EffectiveQuadSubsampling import EffectiveSubsampling
import effective_quadratures.Utils as utils
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
    return np.exp(x[0])

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 5
    derivative_flag = 0 # derivative flag
    min_value = -3.0
    max_value = 1.0
    q_parameter = 1.00

    # Decide on the polynomial basis. We recommend total order or hyperbolic cross
    # basis terms. First we create an index set object
    #hyperbolic_basis = IndexSet("total order", [order-1, order-1, order-1, order-1])
    hyperbolic_basis = IndexSet("tensor grid", [order-1])
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_basis)
    index_elements = IndexSet.getIndexSet(hyperbolic_basis)

    # The "UQ" parameters
    uq_parameters = []
    uniform_parameter = PolynomialParam("Uniform", min_value, max_value, [], [] , derivative_flag, order)
    uq_parameters.append(uniform_parameter)


    # Define the EffectiveSubsampling object and get "A"
    effectiveQuads = EffectiveSubsampling(uq_parameters, hyperbolic_basis, derivative_flag)
    A, pts, W, not_used = EffectiveSubsampling.getAsubsampled(effectiveQuads, maximum_number_of_evals)
    b = W * np.mat(utils.evalfunction(pts, fun))
    xn = matrix.solveLeastSquares(A, b)
    mean, variance = stats.compute_mean_variance(xn, index_elements)
    print mean, variance

    uqProblem = PolyParent(uq_parameters, "tensor grid")
    x_full, i, f = PolyParent.getCoefficients(uqProblem, fun)
    print x_full[0,0]


main()
