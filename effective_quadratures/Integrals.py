#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
from EffectiveQuadSubsampling import EffectiveSubsampling
import Utils as utils
import MatrixRoutines as mat
import numpy as np
"""
    Integration utilities.
    Technically, we should just assume the user wants to integrate over a Uniform
    weight and then compute the integral accordingly.
"""

# By default this routine uses a hyperbolic cross space
def effectivelySubsampledGrid(parameter_ranges, function):

    dimensions = len(listOfParameters)
    no_of_pts = []
    for u in range(0, dimensions):
        no_of_pts.append(int(listOfParameters[u].order) )

    # Default value of the hyperbolic cross parameter
    q = 0.75
    hyperbolic_cross = IndexSet("hyperbolic cross", no_of_pts, q)
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_cross)
    effectiveQuads = EffectiveSubsampling(listOfParameters, hyperbolic_cross, 0)
    A, esquad_pts, W, not_used = EffectiveSubsampling.getAsubsampled(effectiveQuads, maximum_number_of_evals)
    A, normalizations = mat.rowNormalize(A)
    b = W * np.mat(utils.evalfunction(esquad_pts, function))
    b = np.dot(normalizations, b)
    xn = mat.solveLeastSquares(A, b)
    integral_esq = xn[0] * 2**dimensions

    return integral_esq, maximum_number_of_evals, esquad_pts

def sparseGrid(listOfParameters, indexSet):

    # Get the number of parameters
    dimensions = len(listOfParameters)

    # Get the sparse index set attributes
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
        tensorObject = PolyParent(listOfParameters, method="tensor grid")
        points, weights = PolyParent.getPointsAndWeights(tensorObject, orders[i,:] )

        # Multiply weights by constant 'a':
        weights = weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )

    #points_store, unique_indices = utils.removeDuplicates(points_store)
    #return points_store, weights_store[unique_indices]
    return points_store, weights_store

def tensorGrid(listOfParameters, indexSet=None):

    # Get the tensor indices
    dimensions = len(listOfParameters)
    max_orders = []
    if indexSet is None:
        for u in range(0, dimensions):
            max_orders.append(int(listOfParameters[u].order) )
    else:
        max_orders = IndexSet.getMaxOrders(indexSet)

    # Call the gaussian quadrature routine
    tensorObject = PolyParent(listOfParameters, method="tensor grid")
    points, weights = PolyParent.getPointsAndWeights(tensorObject)

    return points, weights

# Returns the overall scaling factor associated with the uqParameters depending
# on their distribution type and their ranges!
def scaleWeights(listOfParameters):
    factor = 0
    dimensions = len(listOfParameters)
    for k in range(0, dimensions):
        if(listOfParameters[k].param_type == 'Uniform' or listOfParameters[k].param_type == 'Beta' ):
            factor = (listOfParameters[k].upper_bound - listOfParameters[k].lower_bound) + factor

    # Final check.
    if factor == 0:
        factor = 1

    return factor
