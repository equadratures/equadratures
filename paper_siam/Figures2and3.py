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

    Plots for:
    "Effective Quadrature Subsampling for Least Squares Polynomial Approximations"
    Seshadri, P., Narayan, A., Mahadevan, S.

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri

    In this figure we plot the error in the pseudospectral coefficients
    for a 2D tensor grid using basis terms from a 2D total order basis set.
    Here we compare randomized quadrature with our technique. 
    
    
    Wrap this up today!!!
"""
# Simple analytical function
def fun(x):
    return np.exp(x[0] + x[1])

def main():

    # Inputs
    order_in_each_direction = 50
    derivative_flag = 0 # derivative flag
    min_value, max_value = -1, 1
    q_parameter = 0.7

    # We use a hyperbolic cross basis
    hyperbolic_basis = IndexSet("total order", [order_in_each_direction, order_in_each_direction])
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_basis)

    # Uniform parameters between [-1,1]
    uq_parameters = []
    uniform_parameter = PolynomialParam("Uniform", min_value, max_value, [], [] , derivative_flag, order)
    uq_parameters.append(uniform_parameter)
    uq_parameters.append(uniform_parameter)

    # Define the EffectiveSubsampling object and get "A"
    effectiveQuads = EffectiveSubsampling(uq_parameters, hyperbolic_basis, derivative_flag)
    A, pts = EffectiveSubsampling.getAs(effectiveQuads, maximum_number_of_evals)


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


# Then create the A matrix using a subset of columns
# Compute A and C matrices and solve the full least squares problem
A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)
b = fun(gaussPoints, derivative_flag, error_flag)

# Normalize these!
Aweighted , NormFactor = matrix.rowNormalize(A)
bweighted = np.dot(NormFactor, b)

# "REAL" solution
x_true = matrix.solveLeastSquares(Aweighted, bweighted)

# Get the function values at ALL points!
function_values = fun(gaussPoints, derivative_flag, error_flag)

for basis_subsamples in range(2,highest_order):
    for quadrature_subsamples in range(2,highest_order):

        # Now compute the "optimal" subsamples from this grid!
        P = matrix.QRColumnPivoting( A[:, 0 : quadrature_subsamples] )
        optimal = P[ 0 : quadrature_subsamples]

        # Now take the first "evaluations_user_can_afford" rows from P
        Asquare = A[optimal, 0 : basis_subsamples]
        bsquare = b[optimal]
        rows, cols = Asquare.shape

        # Normalize these!
        Asquare, smallNormFactor = matrix.rowNormalize(Asquare)
        bsquare = np.dot(smallNormFactor, bsquare)


        # Solve least squares problem only if rank is not degenrate!
        if(np.linalg.matrix_rank(Asquare) == cols):
            # Solve the least squares problem
            x = matrix.solveLeastSquares(Asquare, bsquare)
            store_error[basis_subsamples,quadrature_subsamples] = np.linalg.norm( x - x_true[0:basis_subsamples])

            # Compute the condition numbers of these matrices!
            store_cond[basis_subsamples, quadrature_subsamples] = np.linalg.cond(Asquare)
