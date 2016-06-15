#!/usr/bin/env python
from PolyParams import PolynomialParam
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
class EffectiveQuadrature(object):

    def __init__(self, uq_parameters, method, level=None, growth_rule=None,index_sets=None):

        self.uq_parameters = uq_parameters
        self.method = method

        # Check for the levels (only for sparse grids)
        if level is None:
            self.level = []
        else:
            self.level = level

        # Check for the growth rule (only for sparse grids)
        if growth_rule is None:
            self.growth_rule = []
        else:
            self.growth_rule = growth_rule

        # Here we set the index sets!
        if index_sets is None:

            # Determine the highest orders for a tensor grid
            highest_orders = []
            for i in range(0, len(uq_parameters)):
                highest_orders.append(uq_parameters[i].order)

            if(method == "tensor grid" or method == "Tensor grid"):
                indexObject = IndexSet(method, highest_orders)
                self.index_sets = indexObject

            if(method == "sparse grid" or method == "Sparse grid"):
                indexObject = IndexSet(method, highest_orders, level, growth_rule)
                self.index_sets = indexObject

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    get() methods
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    def getRandomizedTensorGrid(self):
        return getSubsampledGaussianQuadrature(self)

    def getMultivariatePoly(self, points):
        return getMultiOrthoPoly(self, points)

    def getMultivariateA(self, points):

        # Preliminaries
        indices = self.indexsets
        no_of_indices, dimensions = indices.shape
        A_univariate = {}
        total_points = len(points[:,0])

        # Assuming we have no derivatives?
        for i in range(0, dimensions):
            P, M = PolynomialParam.getOrthoPoly(self.uq_parameters[i], points[:,i])
            A_univariate[i] = P
            local_rows, local_cols = A_univariate[i].shape

        # Now based on the index set compute the big ortho-poly matrix!
        A_multivariate = np.zeros((no_of_indices, total_points))
        for i in range(0, no_of_indices):
            temp = np.ones((1,total_points))
            for j in range(0, dimensions):
                A_multivariate[i, :] =  A_univariate[j][indices[i,j], :] * temp
                temp = A_multivariate[i, :]

        # Take the transpose!
        A_multivariate = A_multivariate.T
        return A_multivariate

    def getCoefficients(self, function):
        if self.method == "tensor grid" or self.method == "Tensor grid":
            return getPseudospectralCoefficients(self.uq_parameters, function)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            return getSparsePseudospectralCoefficients(self, function)

    def getPointsAndWeights(self, *argv):
        if self.method == "tensor grid" or self.method == "Tensor grid":
            return getGaussianQuadrature(self.uq_parameters)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            indexSets = self.index_sets
            if len(sys.argv) > 0:
                level =  argv[0]
                growth_rule = argv[1]
            else:
                error_function('ERROR: To compute the points of a sparse grid integration rule, level and growth rule are required.')

            level = self.level
            growth_rule = self.growth_rule
            sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
            return sparsegrid(self.uq_parameters, self.index_sets, level, growth_rule)

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                            PRIVATE FUNCTIONS

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

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
