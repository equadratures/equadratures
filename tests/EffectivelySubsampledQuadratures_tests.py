#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
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
    return np.exp(x[0] + x[1])

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 8
    derivative_flag = 0 # derivative flag
    min_value, max_value = 0, 1
    q_parameter = 0.7

    # Decide on the polynomial basis. We recommend total order or hyperbolic cross
    # basis terms. First we create an index set object
    hyperbolic_basis = IndexSet("hyperbolic cross", [order, order], q_parameter)
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_basis)

    # The "UQ" parameters
    uq_parameters = []
    uniform_parameter = PolynomialParam("Uniform", min_value, max_value, [], [] , derivative_flag, order)
    uq_parameters.append(uniform_parameter)
    uq_parameters.append(uniform_parameter)

    # Define the EffectiveSubsampling object and get "A"
    effectiveQuads = EffectiveSubsampling(uq_parameters, hyperbolic_basis, derivative_flag)
    A, pts = EffectiveSubsampling.getAs(effectiveQuads, maximum_number_of_evals)

    print 'Dimensions of big A'
    print len(A)
    print len(A[0,:])

    """
    ------------------------------------------------------------------------

    Solving the effective quadratures problem!

    ----------------------------------------------------------------------------
    """
    # Step 1 - QR column pivoting
    P = matrix.QRColumnPivoting(A.T)
    print P
    #print P2
    effective = P[ 0 : maximum_number_of_evals]

    # Step 2 - Subsampling
    Asquare = A[effective, :]
    bsquare = utils.evalfunction(pts[effective], fun)

    # Step 3 - Normalize
    Asquare, smallNormFactor = matrix.rowNormalize(Asquare)
    bsquare = np.dot(smallNormFactor, bsquare)

    # Step 4 - Solve the least squares problem
    xapprox = matrix.solveLeastSquares(Asquare, bsquare)

    """
    ------------------------------------------------------------------------

    Solving the tensor grid least squares problem!

    ----------------------------------------------------------------------------
    """
    # Get evaluations at all points!
    b = utils.evalfunction(pts, fun)

    # Normalize
    Abig, NormFactor = matrix.rowNormalize(A)
    bbig = np.dot(NormFactor, b)

    # Now let's solve the least squares problem:
    xfull = matrix.solveLeastSquares(Abig, bbig)

    # Display Output
    print xapprox
    print xfull

main()
