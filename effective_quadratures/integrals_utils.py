#!/usr/bin/env python
"""Utilities for computing integrals of functions"""
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
from EffectiveQuadSubsampling import EffectiveSubsampling

# A tensor grid routine!
def tensorgrid(uq_parameters, function=None):
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
    # Get the tensor indices
    dimensions = len(parameter_ranges)
    parameters = []
    for i in range(0, dimensions):
        parameter = PolynomialParam("Uniform", parameter_ranges[i][0], parameter_ranges[i][1], 0, 0, 0, orders[i])
        parameters.append(parameter)

    # Call the gaussian quadrature routine
    tensorObject = PolyParent(parameters, method="tensor grid")
    points, weights = PolyParent.getPointsAndWeights(tensorObject)
    tensor_int = np.mat(weights) * utils.evalfunction(points, function)

    # Because the uniform weight is defined over [-1,1]
    tensor_int = (tensor_int)/(2**dimensions)
    for i in range(0, dimensions):
        tensor_int = tensor_int * (parameter_ranges[i][1] - parameter_ranges[i][0])

    return tensor_int[0,0], points

def sparseGrid(parameter_ranges,  level, growth_rule, function):
    
    # Get the number of parameters
    dimensions = len(parameter_ranges)
    parameters = []
    for i in range(0, dimensions):
        parameter = PolynomialParam("Uniform", parameter_ranges[i][0], parameter_ranges[i][1], 0, 0, 0, 3)
        parameters.append(parameter)

    # Get the sparse index set attributes
    indexSet = IndexSet("sparse grid", [], level, growth_rule, dimensions)
    sparse_index, a , sg_set = IndexSet.getIndexSet(indexSet)
    rows = len(sparse_index)

    # Get this into an array
    orders = np.zeros((rows, dimensions))
    points_store = []
    weights_store = []
    factor = 1

    for i in range(0, rows):

        # loop through the dimensions
        for j in range(0, dimensions):
            orders[i,j] = np.array(sparse_index[i][j])

        # points and weights for each order~
        tensorObject = PolyParent(parameters, method="tensor grid")
        points, weights = PolyParent.getPointsAndWeights(tensorObject, orders[i,:] )

        # Multiply weights by constant 'a':
        weights = weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )
    sparse_int = np.mat(weights_store) * utils.evalfunction(points_store, function)

    # Because the uniform weight is defined over [-1,1]
    sparse_int = (sparse_int)/(2**dimensions)
    for i in range(0, dimensions):
        sparse_int = sparse_int * (parameter_ranges[i][1] - parameter_ranges[i][0])

    point_store = utils.removeDuplicates(points_store)
    return sparse_int[0,0], points_store

    
#
# By default this routine uses a hyperbolic cross space
# Assume that the inputs are given as tuples: parameters = [(-1,1), (0,0), ...]
def effectivelySubsampledGrid(parameter_ranges, orders, q, function):

    dimensions = len(parameter_ranges)
    parameters = []
    for i in range(0, dimensions):
        parameter = PolynomialParam("Uniform", parameter_ranges[i][0], parameter_ranges[i][1], 0, 0, 0, orders[i])
        parameters.append(parameter)

    # Define the hyperbolic cross
    hyperbolic_cross = IndexSet("hyperbolic cross", orders, q)
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_cross)
    effectiveQuads = EffectiveSubsampling(parameters, hyperbolic_cross, 0)
    A, esquad_pts, W, not_used = EffectiveSubsampling.getAsubsampled(effectiveQuads, maximum_number_of_evals)
    A, normalizations = mat.rowNormalize(A)
    b = W * np.mat(utils.evalfunction(esquad_pts, function))
    b = np.dot(normalizations, b)
    xn = mat.solveLeastSquares(A, b)
    integral_esq = xn[0]
    for i in range(0, dimensions):
        integral_esq  = integral_esq  * (parameter_ranges[i][1] - parameter_ranges[i][0])
    return integral_esq[0], esquad_pts



