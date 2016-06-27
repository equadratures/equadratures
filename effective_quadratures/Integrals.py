#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import numpy as np
"""
    Integration utilities.
"""
def sparseGrid(uqStructure, indexSet):

    # Get the sparse indices
    uq_para = uqStructure.uq_parameters
    indexSets = uqStructure.index_sets
    level = uqStructure.level
    growth_rule = uqStructure.growth_rule
    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, sg_set_full = IndexSet.getIndexSet(indexSets)



    dimensions = len(uq_parameters)
    sparse_index, a , sg_set = IndexSet.getIndexSet(indexSetObject)

    # Compute the corresponding Gauss quadrature points and weights
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
        points, weights = PolyParent.getPointsAndWeights(uqStructure)

        # Multiply weights by constant 'a':
        weights = scaleWeights(uqParameters) * weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )

    #points_store, unique_indices = utils.removeDuplicates(points_store)
    #return points_store, weights_store[unique_indices]
    return points_store, weights_store

def tensorGrid(uqStructure):

    # Get the tensor indices
    dimensions = len(uq_parameters)
    all_indices = IndexSet.getIndexSet(indexSetObject)

    # Just get the highest order in each direction
    max_orders = []
    for u in range(0, dimensions):
        max_orders.append(int(np.max(all_indices[:,u]) ) )

    # Call the gaussian quadrature routine
    points, weights = PolyParent.getPointsAndWeights(uqParameters, max_orders)

    # Multiply by the factor
    weights = weights * scaleWeights(uqParameters)

    return points, weights

# Returns the overall scaling factor associated with the uqParameters depending
# on their distribution type and their ranges!
def scaleWeights(uqParameters):
    factor = 0
    for k in range(0, dimensions):
        if(uqParameters[i].param_type == 'Uniform' or uqParameters[i].param_type == 'Beta' ):
            factor = (Parameters[k].upper_bound - Parameters[k].lower_bound) + factor

    # Final check.
    if factor == 0:
        factor = 1

    return factor
