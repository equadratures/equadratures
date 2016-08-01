#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import numpy as np
"""
    Integration utilities.
"""
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
