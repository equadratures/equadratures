#!/usr/bin/env python
from parameter import Parameter
from polynomial import Polynomial
from qr_factorization import mgs_pivoting, solveLSQ
from indexset import IndexSet
from utils import error_function, evalfunction
import numpy as np
from scipy.linalg import qr
"""
Pranay Seshadri
ps583@cam.ac.uk
"""
class EffectiveSubsampling(object):

    def __init__(self, uq_parameters, index_set, derivative_flag=None):

        self.uq_parameters = uq_parameters
        self.index_set = index_set

        if derivative_flag is None:
            derivative_flag = 0
        else:
            self.derivative_flag = derivative_flag        

    def getAmatrix(self):
        return getA(self)

    def getAsubsampled(self, maximum_number_of_evals, flag=None):
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag=None)
        return Asquare

    def getPointsToEvaluate(self, maximum_number_of_evals, flag=None):
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag=None)
        return esq_pts

    def solveLeastSquares(self, maximum_number_of_evals, function_values):
        A, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag=None)
        A, normalizations = rowNormalize(A)
        
        # Check if user input is a function or a set of function values!
        if callable(function_values):
            fun_values = evalfunction(esq_pts, function_values)
        else:
            fun_values = function_values
        
        b = W * fun_values
        b = np.dot(normalizations, b)
        x = solveLSQ(A, b)
        return x

# Normalize the rows of A by its 2-norm  
def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization

# A matrix formed by a tensor grid of rows and a user-defined set of columns.
def getA(self):
    stackOfParameters = self.uq_parameters
    polynomial_basis = self.index_set
    dimensions = len(stackOfParameters)
    indices = IndexSet.getIndexSet(polynomial_basis)
    no_of_indices = len(indices)

    # Crate a new PolynomialParam object to get tensor grid points & weights
    polyObject =  Polynomial(stackOfParameters, "tensor grid")
    quadrature_pts, quadrature_wts = Polynomial.getPointsAndWeights(polyObject)

    # Allocate memory for "unscaled points!"
    unscaled_quadrature_pts = np.zeros((len(quadrature_pts), dimensions))
    for i in range(0, dimensions):
        for j in range(0, len(quadrature_pts)):
                if (stackOfParameters[i].param_type == "Uniform"):
                    unscaled_quadrature_pts[j,i] = ((quadrature_pts[j,i] - stackOfParameters[i].lower)/(stackOfParameters[i].upper - stackOfParameters[i].lower))*2.0 - 1.0

                elif (stackOfParameters[i].param_type == "Beta" ):
                    unscaled_quadrature_pts[j,i] = (quadrature_pts[j,i] - stackOfParameters[i].lower)/(stackOfParameters[i].upper - stackOfParameters[i].lower)

    # Ensure that the quadrature weights sum up to 1.0
    quadrature_wts = quadrature_wts/np.sum(quadrature_wts)
    P = np.mat(Polynomial.getMultivariatePolynomial(polyObject, unscaled_quadrature_pts, indices))
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

# The subsampled A matrix based on either randomized selection of rows or a QR column pivoting approach
def getSquareA(self, maximum_number_of_evals, flag=None):

    if flag == "QR" or flag is None:
        option = 1 # default option!
    elif flag == "Random":
        option = 2
    else:
        error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): For the third input choose from either 'QR' or 'Random'")

    A, quadrature_pts, quadrature_wts = getA(self)
    dimension = len(self.uq_parameters)
    m , n = A.shape

    if maximum_number_of_evals < n :
        print 'Dimensions of A prior to subselection:'
        print m, n
        print 'The maximum number of evaluations you requested'
        print maximum_number_of_evals
        error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): The maximum number of evaluations must be greater or equal to the number of basis terms")

    # Now compute the rank revealing QR decomposition of A!
    if option == 1:
        P = mgs_pivoting(A.T)
        #Q, R, P = qr(A.T, pivoting=True)
    else:
        P = np.random.randint(0, len(quadrature_pts) - 1, len(quadrature_pts) - 1 )

    # Now truncate number of rows based on the maximum_number_of_evals
    selected_quadrature_points = P[0:maximum_number_of_evals]
        
    # Form the "square" A matrix.
    Asquare =  getRows(np.mat(A), selected_quadrature_points)
    
    #print np.linalg.cond(Asquare)
    esq_pts = getRows(np.mat(quadrature_pts), selected_quadrature_points)
    esq_wts = quadrature_wts[selected_quadrature_points]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    return Asquare, esq_pts, W, selected_quadrature_points

# Function that returns a submatrix of specific rows
def getRows(A, row_indices):
    
    # Determine the shape of A
    m , n = A.shape

    # Allocate space for the submatrix
    A2 = np.zeros((len(row_indices), n))

    # Now loop!
    for i in range(0, len(A2)):
        for j in range(0, n):
            A2[i,j] = A[row_indices[i], j]

    return A2