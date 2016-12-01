#!/usr/bin/env python
"""Effectively subsampled quadratures for least squares polynomial approximations"""
from parameter import Parameter
from polynomial import Polynomial
from qr import mgs_pivoting, solveLSQ, solve_constrainedLSQ
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
    def __init__(self, uq_parameters, index_set=None, method=None):
        self.uq_parameters = uq_parameters
        dimensions = len(uq_parameters)

        # For increased flexibility, if the index_set is not given, we will assume a tensor grid basis
        if index_set is None:

            # determine the orders!
            orders_to_use = []
            for u in range(0, dimensions):
                orders_to_use.append(np.int(uq_parameters[u].order - 1) )

            # Use the tensor grid option!
            self.index_set = IndexSet("Tensor grid", orders_to_use)    

        else:
            # Now before we set self.index_set = index_set, we check to make sure that
            # the number of basis used is -1 the number of points!
            orders_to_use = []
            count = 0
            for u in range(0, dimensions):
                orders_to_use.append( np.int(uq_parameters[u].order) )
                if orders_to_use[u] <= index_set.orders[u] :
                    count = count + 1
            if count > 0:
                error_function('IndexSet: Basis orders: Ensure that the basis order is always -1 the number of points!')
            
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
        :rtype: numpy matrix
        
        **Sample declaration**
        :: 
            >> eq.getASubsampled()
        """
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
        return Asquare, esq_pts, W, points
    
    def getCmatrix(self):
        """
        put some comments here!
        """
        stackOfParameters = self.uq_parameters
        polynomial_basis = self.index_set
        dimensions = len(stackOfParameters)
        polyObject_for_basis = Polynomial(stackOfParameters, polynomial_basis) 
        points, weights = polyObject_for_basis.getPointsAndWeights()
        not_used, C = polyObject_for_basis.getMultivariatePolynomial(points)
        Cfull = cell2matrix(C)
        return Cfull

    def getCsubsampled(self, maximum_number_of_evals):
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
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
        not_used, C = polyObject_for_basis.getMultivariatePolynomial(esq_pts)
        Cfull = cell2matrix(C)
        return Cfull

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
        Asquare, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
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
        A, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
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
        #print 'Dimensions of A prior to subselection:'
        #print m, n
        #print 'The maximum number of evaluations you requested'
        #print maximum_number_of_evals

        # Now if the derivative flag option is activated, we do not raise an error. Otherwise an error is raised!
        if self.uq_parameters[0].derivative_flag is None:
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

def cell2matrix(G):
    dimensions = len(G)
    G0 = G[0] # Which by default has to exist!
    C0 = G0.T
    rows, cols = C0.shape
    BigC = np.zeros((dimensions*rows, cols))
    counter = 0
    for i in range(0, dimensions):
        K = G[i].T
        for j in range(0, rows):
            for k in range(0,cols):
                BigC[counter,k] = K[j,k]
            counter = counter + 1 
    BigC = np.mat(BigC)
    return BigC

































    def solveLeastSquaresWithGradients(self, maximum_number_of_evals, function_values, gradient_values):
        """
        Returns the coefficients for the effectively subsampled quadratures least squares problem. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param integer maximum_number_of_evals: The maximum number of evaluations the user would like. This value has to be atleast equivalent to the
            total number of basis terms of the index set.    
        :param callable function_values: A function call to the simulation model, that takes in d inputs and returns one output. If users know the 
            quadrature subsamples required, they may also input all the simulation outputs as a single ndarray.     
        :param callable gradient_values: A function call to the simulation model, that takes in d inputs and returns the dx1 gradient vector at those inputs.
            If the user knows the quadrature subsampled required, they may also input all the simulation gradients as an nd array. 
        :return: x, the coefficients of the least squares problem.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> x = eq.solveLeastSquares(150, function_call)
        """
        A, esq_pts, W, points = getSquareA(self, maximum_number_of_evals)
        A, normalizations = rowNormalize(A)     
        C = self.getCsubsampled(esq_pts)
        
        # Check if user input is a function or a set of function values!
        if callable(function_values):
            fun_values = evalfunction(esq_pts, function_values)
        else:
            fun_values = function_values
        
        if callable(gradient_values):
            grad_values = evalfunction(esq_pts, gradient_values)
        else:
            grad_values = gradient_values

        # Weight and row normalize function values!
        b = W * fun_values
        b = np.dot(normalizations, b)

        # Weight and row normalize gradient values!
        # Assume that the gradient values are given as a matrix
        # First check if the dimensions make sense...then weight them
        # Then send them to the lsqr routine...

        # Now the gradient values will usually be arranged as a N-by-d matrix,
        # where N are the number of points and d is the number of dimensions.
        # This needs to be changed into a single vector
        p, q = grad_values.shape
        d_vec = np.zeros((p*q,1))
        counter = 0
        for j in range(0,q):
            for i in range(0,p):
                d_vec[counter] = grad_values[i,j]
                counter = counter + 1

        # Now solve the constrained least squares problem
        return solve_constrainedLSQ(A, b, C, d_vec)