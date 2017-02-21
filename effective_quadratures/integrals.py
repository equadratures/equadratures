#!/usr/bin/env python
"""Utilities for computing integrals of functions"""
from parameter import Parameter
from polynomial import Polynomial
from indexset import IndexSet
from effectivequads import EffectiveSubsampling
import numpy as np
#****************************************************************************
# Functions to code:
#    
# 1. Faster Gauss quadrature rules (see work by Alex Townsend & Nick Hale)
# 2. Sparse grid quadrature rules with different growth rules
# 3. Spherical quadrature rules
# 4. Padua quadrature rules -- add 2D!
#****************************************************************************
class EQ_Integration(object):
    """
    Quadrature rules for integrating a function over the interval [-1,1]^d.
    For multivariate quadrature rules on non-tensorial domains, this routine will use effective subsampling
    to compute appropriate points and weights.

    :param int order: Maximum order along each direction
    :param basis: For multivariate domains, user may select 'Total order', 'Hyperbolic cross' or 'Euclidean'. 
        The hyperbolic cross parameter is set by default to 0.5.
    :param method: Determines the type of quadrature rule to be used. Available options are: 
        Gauss-Legendre or Clenshaw-Curtis.
    :param coords: The domain over which the integral is to be computed. Options are, 'Cartesian' (default) or 'Spherical'

    **Sample declarations** 
    ::
        # Uniform distribution with 5 points on [-2,2]
        >> Parameter(points=5, lower=-2, upper=2, param_type='Uniform')

    """
    def __init__(self, order, basis=None, method=None, coords=None):
        
        self.order = order
        dimension = len(order)

        if  
        self.basis = basis

        if basis is None
        p, w = 
        self.points = 



def getCartesianPointsAndWeights(order, method):
    if method is 'Gauss':
        p, w = getGauss(order)
    elif method is 'Clenshaw-Curtis':
        p, w = getClenshawCurtis(order)
    elif method is 'Padua':
        p, w = getPadua(order)
    return 0
         
    
def tensorgrid(stackOfParameters, function=None):
    """
    Computes a tensor grid of quadrature points based on the distributions for each Parameter in stackOfParameters 

    :param Parameter array stackOfParameters: A list of Parameter objects
    :param callable function: The function whose integral needs to be computed. Can also be input as an array of function values at the
        quadrature points. If the function is given as a callable, then this routine outputs the integral of the function and an array of
        the points at which the function was evaluated at to estimate the integral. These are the quadrature points. In case the function is
        not given as a callable (or an array, for that matter), then this function outputs the quadrature points and weights. 
      
    :return: tensor_int: The tensor grid approximation of the integral
    :rtype: double
    :return: points:  The quadrature points
    :rtype: numpy ndarray
    :return: weights: The quadrature weights
    :rtype: numpy ndarray

    **Notes**
    For further details on this routine, see Polynomial.getPointsAndWeights()

    """
    # Determine the index set to be used!
    dimensions = len(stackOfParameters)
    orders = []
    flags = []
    uniform = 1
    not_uniform = 0
    for i in range(0, dimensions):
        orders.append(stackOfParameters[i].order)
        if stackOfParameters[i].param_type is 'Uniform':
            flags.append(uniform)
        else:
            flags.append(not_uniform)

    tensor = IndexSet('Tensor grid', orders)
    polyObject = Polynomial(stackOfParameters, tensor)

    # Now compute the points and weights
    points, weights = polyObject.getPointsAndWeights()
    
    # For normalizing!
    for i in range(0, dimensions):
        if flags[i] == 0:
            weights  = weights
        elif flags[i] == 1:
            weights = weights * (stackOfParameters[i].upper - stackOfParameters[i].lower )
            weights = weights/(2.0)

    # Now if the function is a callable, then we can compute the integral:
    if function is not None and callable(function):
        tensor_int = np.mat(weights) * evalfunction(points, function)
        return tensor_int, points
    else:
        return points, weights

def sparsegrid(stackOfParameters, level, growth_rule, function=None):
    """
    Computes a sparse grid of quadrature points based on the distributions for each Parameter in stackOfParameters 

    :param Parameter array stackOfParameters: A list of Parameter objects
    :param integer level: Level parameter of the sparse grid integration rule
    :param string growth_rule: Growth rule for the sparse grid. Choose from 'linear' or 'exponential'.
    :param callable function: The function whose integral needs to be computed. Can also be input as an array of function values at the
        quadrature points. If the function is given as a callable, then this routine outputs the integral of the function and an array of
        the points at which the function was evaluated at to estimate the integral. These are the quadrature points. In case the function is
        not given as a callable (or an array, for that matter), then this function outputs the quadrature points and weights. 
      
    :return: sparse_int: The sparse grid approximation of the integral
    :rtype: double
    :return: points:  The quadrature points
    :rtype: numpy ndarray
    :return: weights: The quadrature weights
    :rtype: numpy ndarray

    """
    # Determine the index set to be used!
    dimensions = len(stackOfParameters)
    orders = []
    flags = []
    uniform = 1
    not_uniform = 0
    for i in range(0, dimensions):
        orders.append(stackOfParameters[i].order)
        if stackOfParameters[i].param_type is 'Uniform':
            flags.append(uniform)
        else:
            flags.append(not_uniform)
        
    # Call the sparse grid index set
    sparse = IndexSet('Sparse grid', level=level, growth_rule=growth_rule, dimension=dimensions)
    sparse_index, sparse_coeffs, sparse_all_elements =  sparse.getIndexSet()

    # Get this into an array
    rows = len(sparse_index)
    orders = np.zeros((rows, dimensions))
    points_store = []
    weights_store = []
    factor = 1
    

    # Now get the tensor grid for each sparse_index
    for i in range(0, rows):

        # loop through the dimensions
        for j in range(0, dimensions):
            orders[i,j] = np.array(sparse_index[i][j])

        # points and weights for each order~
        tensor = IndexSet('Tensor grid', orders)
        polyObject = Polynomial(stackOfParameters, tensor)
        points, weights = polyObject.getPointsAndWeights(orders[i,:] )

        # Multiply weights by constant 'a':
        weights = weights * sparse_coeffs[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )
    
    # For normalizing!
    for i in range(0, dimensions):
        if flags[i] == 0:
            weights_store  = weights_store
        elif flags[i] == 1:
            weights_store = weights_store * (stackOfParameters[i].upper - stackOfParameters[i].lower )
            weights_store = weights_store/(2.0)

    # Now if the function is a callable, then we can compute the integral:
    if function is not None and callable(function):
        sparse_int = np.mat(weights_store) * evalfunction(points_store, function)
        point_store = removeDuplicates(points_store)
        return sparse_int, points_store
    else:
        point_store = removeDuplicates(points_store)
        return points_store, weights_store
    
def effectivequadratures(stackOfParameters, q_parameter, function):
    """
    Computes an approximation of the integral using effective-quadratures; this routine uses least squares to estimate the integral.

    :param Parameter array stackOfParameters: A list of Parameter objects
    :param double q_parameter: By default, this routine uses a hyperbolic polynomial basis where the q_parameter (value between 0.1 and 1.0) adjusts the number
        of basis terms to be used for solving the least squares problem.
    :param callable function: The function whose integral needs to be computed. Can also be input as an array of function values at the
        quadrature points. The function must be provided either as a callable or an array of values for this routine to work. 
      
    :return: integral_esq: The effective quadratures approximation of the integral
    :rtype: double
    :return: points:  The quadrature points
    :rtype: numpy ndarray

    """

    # Determine the index set to be used!
    dimensions = len(stackOfParameters)
    orders = []
    flags = []
    uniform = 1
    not_uniform = 0
    for i in range(0, dimensions):
        orders.append(int(stackOfParameters[i].order - 1) )
        if stackOfParameters[i].param_type is 'Uniform':
            flags.append(uniform)
        else:
            flags.append(not_uniform)

    # Define the hyperbolic cross
    hyperbolic = IndexSet('Hyperbolic basis', orders=orders, q=q_parameter)
    maximum_number_of_evals = hyperbolic.getCardinality()
    effectiveQuads = EffectiveSubsampling(stackOfParameters, hyperbolic)

    # Call the effective subsampling object
    points = effectiveQuads.getEffectivelySubsampledPoints(maximum_number_of_evals)
    xn = effectiveQuads.solveLeastSquares(maximum_number_of_evals, function)
    integral_esq = xn[0]

    # For normalizing!
    for i in range(0, dimensions):
        if flags[i] == 0:
            integral_esq  = integral_esq
        elif flags[i] == 1:
            integral_esq = integral_esq * (stackOfParameters[i].upper - stackOfParameters[i].lower )
    
    return integral_esq[0], points

# Evaluate the function (above) at certain points
def evalfunction(points, function):
    function_values = np.zeros((len(points), 1))

    # For loop through all the points
    for i in range(0, len(points)):
        function_values[i,0] = function(points[i,:])

    return function_values

# Simple function that removes duplicate rows from a matrix and returns the
# deleted row indices
def removeDuplicates(a):
    order = np.lexsort(a.T)
    a = a[order]
    indices = []
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    # Save the indices
    for i in range(0, len(ui)):
        if(ui[i] == bool(1)):
            indices.append(i)
    return a[ui], indices
