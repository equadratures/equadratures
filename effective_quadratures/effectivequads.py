#!/usr/bin/env python
"""Effectively subsampled quadratures for least squares polynomial approximations"""
from parameter import Parameter
from polynomial import Polynomial
from qr import mgs_pivoting, solveLSQ
from indexset import IndexSet
from utils import error_function, evalfunction
import numpy as np

class EffectiveSubsampling(object):
    """
    This class defines an EffectiveSubsampling object. Below are details of its constructor.

    :param array of Parameters uq_parameters: A list of Parameters
    :param IndexSet index_set: The index set corresponding to a polynomial basis
    :param string method: Subsampling strategy; options include: 'QR', which is the default option and 'Random'. See this `paper <https://arxiv.org/abs/1601.05470>`_, for further details. 
   
    **Sample declarations** 
    ::
        
        >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
        >> I = IndexSet('Total order' [3, 3, 3])
        >> eq = EffectiveSubsampling([var1, var1], I)
    """
    # Constructor
    def __init__(self, uq_parameters, index_set, method=None):
        self.uq_parameters = uq_parameters
        self.index_set = index_set
        if method is not None:
            self.method = method
        else:
            self.method = 'QR'
            
    def getAmatrix(self):
        """
        Returns a matrix of multivariate orthogonal polynomials evaluated at a tensor grid of quadrature points.

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :return: A, an m-by-k matrix where m is the cardinality of the index set used to define the EffectiveSubsampling object
            and k are the number of tensor grid quadrature points formed by the order prescribed when defining each Parameter in 
            the EffectiveSubsampling object
        :rtype: numpy matrix

        **Sample declaration**
        :: 
            >> eq.getAMatrix()
        """
        return getA(self)

    def getAsubsampled(self, maximum_number_of_evals):
        """
        Returns a matrix of multivariate orthogonal polynomials evaluated at a subsample of the tensor grid of quadrature points.

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param integer maximum_number_of_evals: The maximum number of evaluations the user would like. This value has to be atleast equivalent to the
            total number of basis terms of the index set.
        :return: A, an m-by-k matrix where m is the cardinality of the index set used to define the EffectiveSubsampling object
            and k are the number of subsamples given by the integer maximum_number_of_evals.

        **Sample declaration**
        :: 
            >> eq.getASubsampled()
        """
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
        return Asquare
    
    def getCsubsampled(self, quadrature_subsamples):
        """
        Returns a matrix of multivariate derivative orthogonal polynomials evaluated at a set of quadrature points.

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param ndarray quadrature_subsamples: The quadrature points are which the matrix of derivative orthogonal polynomials must be evaluated at.
        :return: C, a cell with d matrices of size m-by-l matrix where m is the cardinality of the index set used to define the EffectiveSubsampling object
            and l is the number of points in quadrature_subsamples. The integer d represents the dimensionality of the problem, equivalent to the number of
            parameters defined when initializing the constructor.
        :rtype: numpy matrix

        **Sample declaration**
        :: 
            >> eq.getCSubsampled()
        """
        stackOfParameters = self.uq_parameters
        polynomial_basis = self.index_set
        dimensions = len(stackOfParameters)
        polyObject_for_basis = Polynomial(stackOfParameters, polynomial_basis) 
        not_used, C = polyObject_for_basis.getMultivariatePolynomial(quadrature_subsamples)
        return C

    def getEffectivelySubsampledPoints(self, maximum_number_of_evals, flag=None):
        """
        Returns the effectively subsampled quadrature points. See this `paper <https://arxiv.org/abs/1601.05470>`_, for further details. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param integer maximum_number_of_evals: The maximum number of evaluations the user would like. This value has to be atleast equivalent to the
            total number of basis terms of the index set.        
        :return: esq_pts, a maximum_number_of_evals-by-d matrix of quadrature points, where d represents the dimensionality of the problem.
        :rtype: numpy matrix

        **Sample declaration**
        :: 
            >> eq.getEffectivelySubsampledPoints(30)
        """
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag)
        return esq_pts

    def solveLeastSquares(self, maximum_number_of_evals, function_values):
        """
        Returns the coefficients for the effectively subsampled quadratures least squares problem. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param integer maximum_number_of_evals: The maximum number of evaluations the user would like. This value has to be atleast equivalent to the
            total number of basis terms of the index set.    
        :param callable function_values: A function call to the simulation model, that takes in d inputs and returns one output. If users know the 
            quadrature subsamples required, they may also input all the simulation outputs as a single ndarray.     
        :return: x, the coefficients of the least squares problem.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> x = eq.solveLeastSquares(150, function_call)
        """
        A, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag)
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

    def solveLeastSquaresWithGradients(self, maximum_number_of_evals, function_values, grad_values):
        """
        Returns the coefficients for the effectively subsampled quadratures least squares problem. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param integer maximum_number_of_evals: The maximum number of evaluations the user would like. This value has to be atleast equivalent to the
            total number of basis terms of the index set.    
        :param callable function_values: A function call to the simulation model, that takes in d inputs and returns one output. If users know the 
            quadrature subsamples required, they may also input all the simulation outputs as a single ndarray.     
        :return: x, the coefficients of the least squares problem.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> x = eq.solveLeastSquares(150, function_call)
        """
        A, esq_pts, W, points = getSquareA(self, maximum_number_of_evals, flag)
        A, normalizations = rowNormalize(A)     
        C = getSubsampled(self, esq_pts)
        
        return 0


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
    polyObject_for_pts =  Polynomial(stackOfParameters)
    quadrature_pts, quadrature_wts = polyObject_for_pts.getPointsAndWeights()

    polyObject_for_basis = Polynomial(stackOfParameters, polynomial_basis) 

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

    # Now we create another Polynomial object for the basis set!
    polynomial_expansions, no_used = polyObject_for_basis.getMultivariatePolynomial(unscaled_quadrature_pts)
    P = np.mat(polynomial_expansions)
    m, n = P.shape
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

# The subsampled A matrix based on either randomized selection of rows or a QR column pivoting approach
def getSquareA(self, maximum_number_of_evals):
    
    flag = self.method
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
