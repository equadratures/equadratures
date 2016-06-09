#!/usr/bin/python
from PolyParams import PolynomialParam

import PolyMethod as polmeth
import IndexSets as isets
import numpy as np
"""

    Integrals Class

    Notes to self:
    - user provides function, and you return an Integrals
    - user wants quadrature points and weights
    - class basics: integral_type : sparse, tensor ; for sparse -- optional parameters
    - leverage index_sets.py

    Pranay Seshadri
    ps583@cam.ac.uk

"""
def SparseGrid(Parameters, level, growth_rule):

    # Get the sparse indices
    dimensions = len(Parameters)
    sparse_index, a , sg_set = isets.sparse_grid_index_set(dimensions, level, growth_rule)

    # Compute the corresponding Gauss quadrature points and weights
    rows = len(sparse_index)

    # Get this into an array
    orders = np.zeros((rows, dimensions))
    points_store = []
    weights_store = []


    # Ok, now we have to correct for the weights, depending on the right and left
    # bounds of the individual parameters. I'm hardcoding this for Legendre for
    # the moment!
    factor = 0
    for k in range(0, dimensions):
        factor = (Parameters[k].upper_bound - Parameters[k].lower_bound) + factor


    for i in range(0, rows):

        # loop through the dimensions
        for j in range(0, dimensions):
            orders[i,j] = np.array(sparse_index[i][j])

        # points and weights for each order~
        points, weights = polmeth.getGaussianQuadrature(Parameters, orders[i,:])

        # Multiply weights by constant 'a':
        weights = factor * weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            #print points[k,:]
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )

    return points_store, weights_store

def TensorGrid(Parameters, orders):

    # Call the gaussian quadrature routine
    points, weights = polmeth.getGaussianQuadrature(Parameters, orders)

    # Get the weight factor:
    dimensions = len(Parameters)
    factor = 0
    for k in range(0, dimensions):
        factor = (Parameters[k].upper_bound - Parameters[k].lower_bound) + factor

    # Multiply by the factor
    weights = weights * factor

    return points, weights
