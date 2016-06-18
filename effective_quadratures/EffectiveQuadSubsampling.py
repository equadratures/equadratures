#!/usr/bin/env python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import numpy as np
import sys
import MatrixRoutines as mat

"""
    Effectively Subsamples Quadratures Class

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    WARNING: Coding in progress!

"""

# Here we demonstrate our new strategy!
#
# inputs:
# 1. Index set type
# 2. Tensor grid size
# 3. uncertainties
# 4. Plotting
# 5. Read data from file!?
#
# From the orders in uq_parameters construct a tensor grid.
# Use the index set as the basis
# Insert the mutivariate "A" matrix Computation Here
# Call QR column pivoting based on a square matrix construction
#
class EffectiveSubsampling(object):

    def __init__(self, uq_parameters, index_set, derivative_flag):

        self.uq_parameters = uq_parameters
        self.index_set = index_set


    " The A matrix"
    def getA(self, points=None):
        #----------------------------------------------------------------------
        # INPUTS:
        # self: EffectiveQuadrature object
        # points: If user does not want to use default tensor grid of points
        #----------------------------------------------------------------------
        stackOfParameters = self.uq_parameters
        polynomial_basis = self.index_set
        dimensions = len(stackOfParameters)
        indices = IndexSet.getIndexSet(polynomial_basis)
        no_of_indices = len(indices)

        # Crate a new PolynomialParam object to get tensor grid points & weights
        polyObject =  PolyParent(stackOfParameters, "tensor grid")
        quadrature_pts, quadrature_wts = PolyParent.getPointsAndWeights(polyObject)

        # Allocate space for each of the univariate matrices!
        A_univariate = {}
        total_points = len(quadrature_pts[:,0])

        " Assuming we have no derivatives"
        for i in range(0, dimensions):

            # Create a polynomial object!
            N = self.uq_parameters[i].order + 1
            P, M = PolynomialParam.getOrthoPoly(self.uq_parameters[i], quadrature_pts[:,i], N)
            A_univariate[i] = P
            local_rows, local_cols = A_univariate[i].shape

        # Now using the select basis terms, compute multivariate "A". This is
        # a memory intensive operation -- need to figure out a way to handle this.
        A_multivariate = np.zeros((no_of_indices, total_points))
        for i in range(0, no_of_indices):
            temp = np.ones((1,total_points))
            for j in range(0, dimensions):
                A_multivariate[i, :] =  A_univariate[j][indices[i,j], :] * temp
                temp = A_multivariate[i, :]

        # Take the transpose!
        A_multivariate = A_multivariate.T

        return A_multivariate


    def getSquareA(self, maximum_number_of_evals):

"""

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
"""
