#!/usr/bin/env python
"""Effectively subsampled quadratures for least squares polynomial approximations"""
from parameter import Parameter
from polynomial import Polynomial
from qr import qr_MGS, solveLSQ, solveCLSQ
from indexset import IndexSet
from utils import error_function, evalfunction, evalgradients
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

        # Items to set!
        A, quadrature_pts, quadrature_wts = getA(self)
        self.A = A
        self.tensor_quadrature_points = quadrature_pts
        self.tensor_quadrature_weights = quadrature_wts
        if uq_parameters[0].derivative_flag == 1:
            self.C = getC(self)
        else:
            self.C = None
        self.no_of_basis_terms = index_set.getCardinality() 
        self.C_subsampled = None
        self.A_subsampled = None
        self.no_of_evals = None
        self.subsampled_quadrature_points = None
        self.subsampled_quadrature_weights = None # stored as a diagonal matrix??
        self.row_indices = None
        self.dimensions = len(uq_parameters)
    
    def set_no_of_evals(self, no_of_evals):
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

        # Once the user provides the number of evaluations required, we can set a few items!
        self.no_of_evals = no_of_evals
        Asquare, esq_pts, W, row_indices = getSquareA(self)
        self.A_subsampled = Asquare
        self.subsampled_quadrature_points = esq_pts
        self.subsampled_quadrature_weights = W
        self.row_indices = row_indices

        # If the user has turned on the gradient flag!
        if self.C is not None:
            dimensions = len(self.C)
            C0 = self.C[0] # Which by default has to exist!
            C0 = C0.T
            rows, cols = C0.shape
            C_subsampled = np.mat( np.zeros((dimensions*len(row_indices), cols)), dtype='float64')
            counter = 0
            for i in range(0, dimensions):
                temp_matrix = self.C[i].T
                for j in range(0, len(row_indices)):
                    for k in range(0,cols):
                        C_subsampled[counter,k] = temp_matrix[row_indices[j],k]
                    counter = counter + 1 
            self.C_subsampled = C_subsampled

    def prune(self, number_of_columns_to_delete):  
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
        A = self.A_subsampled
        m, n = A.shape
        A_pruned = A[0:m, 0 : (n - number_of_columns_to_delete)]
        self.A_subsampled = A_pruned
        self.index_set

        # If clause for gradient case!
        if self.C is not None:
            C = self.C_subsampled
            p, q = C.shape
            C_pruned = C[0:p, 0 : (q - number_of_columns_to_delete)]
            self.C_subsampled = C_pruned

        self.no_of_basis_terms = self.no_of_basis_terms - number_of_columns_to_delete
        self.index_set.prune(number_of_columns_to_delete)
    
    def least_no_of_subsamples_reqd(self):
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
        k = 1
        self.set_no_of_evals(1)
        rank = np.linalg.matrix_rank(np.mat( np.vstack([self.A_subsampled, self.C_subsampled]), dtype='float64') )
        while rank < self.no_of_basis_terms:
            k = k + 1
            self.set_no_of_evals(k)
            rank = np.linalg.matrix_rank(np.mat( np.vstack([self.A_subsampled, self.C_subsampled]), dtype='float64') )
        return k  

    def computeCoefficients(self, function_values, gradient_values=None, technique=None):
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
        A, normalizations = rowNormalize(self.A_subsampled)
        
        # Check if user input is a function or a set of function values!
        if callable(function_values):
            fun_values = evalfunction(self.subsampled_quadrature_points, function_values)
        else:
            fun_values = function_values
        
        b = self.subsampled_quadrature_weights * fun_values
        b = np.dot(normalizations, b)
        
        ################################
        # No gradient case!
        ################################
        if gradient_values is None:
            x, cond = solveLSQ(A, b)
        
        ################################
        # Gradient case!
        ################################
        else:
            if callable(gradient_values):
                grad_values = evalgradients(self.subsampled_quadrature_points, gradient_values, 'matrix')
            else:
                grad_values = gradient_values
            
            p, q = grad_values.shape
            d = np.zeros((p*q,1))
            counter = 0
            for j in range(0,q):
                for i in range(0,p):
                    d[counter] = grad_values[i,j]
                    counter = counter + 1
            C = self.C_subsampled

            # Now row normalize the Cs and the ds
            

            if technique is None:
                raise(ValueError, 'A technique must be defined for gradient problems. Choose from stacked, equality or inequality. For more information please consult the detailed user guide.')
            else:
                if technique is 'weighted':
                    C, normalizations = rowNormalize(C)
                    d = np.dot(normalizations, d)
                    x, cond = solveCLSQ(A, b, C, d, technique)
                else:
                    x, cond = solveCLSQ(A, b, C, d, technique)
        
        return x, cond

# Normalize the rows of A by its 2-norm  
def rowNormalize(A):
    rows, cols = A.shape
    row_norms = np.mat(np.zeros((rows, 1)), dtype='float64')
    Normalization = np.mat(np.eye(rows), dtype='float64')
    for i in range(0, rows):
        temp = 0.0
        for j in range(0, cols):
            row_norms[i] = temp + A[i,j]**2
            temp = row_norms[i]
        row_norms[i] = (row_norms[i] * 1.0/np.float64(cols))**(-1)
        Normalization[i,i] = row_norms[i]
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
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

# The subsampled A matrix based on either randomized selection of rows or a QR column pivoting approach
def getSquareA(self):

    flag = self.method
    if flag == "QR" or flag is None:
        option = 1 # default option!
    elif flag == "Random":
        option = 2
    else:
        error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): For the third input choose from either 'QR' or 'Random'")

    A = self.A
    m , n = A.shape

    if self.no_of_evals < n :
        
        # Now if the derivative flag option is activated, we do not raise an error. Otherwise an error is raised!
        if self.uq_parameters[0].derivative_flag is None:
            error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): The maximum number of evaluations must be greater or equal to the number of basis terms")

    # Now compute the rank revealing QR decomposition of A!
    if option == 1:
       Q_notused, R_notused, P = qr_MGS(A.T, pivoting=True)
    else:
        P = np.random.randint(0, len(self.tensor_quadrature_points) - 1, len(self.tensor_quadrature_points) - 1 )

    # Now truncate number of rows based on the maximum_number_of_evals
    selected_quadrature_points = P[0:self.no_of_evals]
        
    # Form the "square" A matrix.
    Asquare = A[selected_quadrature_points, :]
    esq_pts = getRows(np.mat(self.tensor_quadrature_points), selected_quadrature_points)
    esq_wts = self.tensor_quadrature_weights[selected_quadrature_points]
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

def getC(self):
        """
        Returns the full C matrix
        """
        stackOfParameters = self.uq_parameters
        polynomial_basis = self.index_set
        dimensions = len(stackOfParameters)
        polyObject_for_basis = Polynomial(stackOfParameters, polynomial_basis) 
        points, weights = polyObject_for_basis.getPointsAndWeights()
        not_used, C = polyObject_for_basis.getMultivariatePolynomial(points)
        return C