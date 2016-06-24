#!/usr/bin/env python
from PolyParams import PolynomialParam
from IndexSets import IndexSet
import numpy as np
import sys
"""

    Polyparent Class
    Designed to be the parent class to the

    Pranay Seshadri
    ps583@cam.ac.uk

    - Bug in the spam : doesn't show all computed coefficients!

"""
class PolyParent(object):
    """ An index set.
    Attributes:

     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                constructor / initializer
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def __init__(self, uq_parameters, method, level=None, growth_rule=None,index_sets=None):

        self.uq_parameters = uq_parameters
        self.method = method

        # Check for the levels (only for sparse grids)
        if level is None:
            self.level = []
        else:
            self.level = level

        # Check for the growth rule (only for sparse grids)
        if growth_rule is None:
            self.growth_rule = []
        else:
            self.growth_rule = growth_rule

        # Here we set the index sets!
        if index_sets is None:

            # Determine the highest orders for a tensor grid
            highest_orders = []
            for i in range(0, len(uq_parameters)):
                highest_orders.append(uq_parameters[i].order)

            if(method == "tensor grid" or method == "Tensor grid"):
                indexObject = IndexSet(method, highest_orders)
                self.index_sets = indexObject

            if(method == "sparse grid" or method == "Sparse grid"):
                indexObject = IndexSet(method, highest_orders, level, growth_rule)
                self.index_sets = indexObject

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    get() methods
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    def getRandomizedTensorGrid(self):
        return getSubsampledGaussianQuadrature(self)

    def getMultivariatePoly(self, points):
        return getMultiOrthoPoly(self, points)

    def getCoefficients(self, function):
        if self.method == "tensor grid" or self.method == "Tensor grid":
            return getPseudospectralCoefficients(self.uq_parameters, function)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            return getSparsePseudospectralCoefficients(self, function)

    def getPointsAndWeights(self, *argv):
        if self.method == "tensor grid" or self.method == "Tensor grid":
            return getGaussianQuadrature(self.uq_parameters)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            indexSets = self.index_sets
            if len(sys.argv) > 0:
                level =  argv[0]
                growth_rule = argv[1]
            else:
                error_function('ERROR: To compute the points of a sparse grid integration rule, level and growth rule are required.')

            level = self.level
            growth_rule = self.growth_rule
            sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
            return sparsegrid(self.uq_parameters, self.index_sets, level, growth_rule)

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                            PRIVATE FUNCTIONS

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def sparsegrid(uq_parameters, indexSetObject, level, growth_rule):

    # Get the sparse indices
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
        points, weights = getGaussianQuadrature(uq_parameters, orders[i,:])

        # Multiply weights by constant 'a':
        weights = factor * weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )

    points_store, unique_indices = removeDuplicates(points_store)
    return points_store, weights_store[unique_indices]

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

def getSparsePseudospectralCoefficients(self, function):

    # INPUTS
    stackOfParameters = self.uq_parameters
    indexSets = self.index_sets
    level = self.level
    growth_rule = self.growth_rule

    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
    rows = len(sparse_indices)
    cols = len(sparse_indices[0])

    for i in range(0,rows):
        for j in range(0, cols):
            sparse_indices[i,j] = int(sparse_indices[i,j])

    # For storage we use dictionaries
    individual_tensor_coefficients = {}
    individual_tensor_indices = {}
    indices = np.zeros((rows,1))

    for i in range(0,rows):
        orders = sparse_indices[i,:] + 1
        K, I = getPseudospectralCoefficients(self.uq_parameters, function, orders)
        individual_tensor_indices[i] = I
        individual_tensor_coefficients[i] =  K
        indices[i,0] = len(I)

    sum_indices = int(np.sum(indices))
    store = np.zeros((sum_indices, dimensions+1))
    counter = int(0)
    for i in range(0,rows):
        for j in range(0, int(indices[i][0])):
             store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][0][j]
             for d in range(0, dimensions):
                 store[counter,d+1] = individual_tensor_indices[i][j][d]
             counter = counter + 1


    # Now we use a while loop to iteratively delete the repeated elements while summing up the
    # coefficients!
    index_to_pick = 0
    flag = 1
    counter = 0

    rows = len(store)

    final_store = np.zeros((sum_indices, dimensions + 1))
    while(flag != 0):
        # find the repeated indices
        rep = find_repeated_elements(index_to_pick, store)
        coefficient_value = 0.0

        # Sum up all the coefficient values
        for i in range(0, len(rep)):
            actual_index = rep[i]
            coefficient_value = coefficient_value + store[actual_index,0]

        # Store into a new array
        final_store[counter,0] = coefficient_value
        final_store[counter,1::] = store[index_to_pick, 1::]
        counter = counter + 1

        # Delete index from store
        store = np.delete(store, rep, axis=0)

        # How many entries remain in store?
        rows = len(store)
        if rows == 0:
            flag = 0

    indices_to_delete = np.arange(counter, sum_indices, 1)
    final_store = np.delete(final_store, indices_to_delete, axis=0)

    # Now split final store into coefficients and their index sets!
    coefficients = np.zeros((1, len(final_store)))
    for i in range(0, len(final_store)):
        coefficients[0,i] = final_store[i,0]

    # Splitting final_store to get the indices!
    indices = final_store[:,1::]

    # Now just double check to make sure they are all integers
    for i in range(0, len(indices)):
        for j in range(0, dimensions):
            indices[i,j] = int(indices[i,j])

    return coefficients, indices

# Tensor grid pseudospectral method
def getPseudospectralCoefficients(stackOfParameters, function, additional_orders=None):

    dimensions = len(stackOfParameters)
    q0 = [1]
    Q = []
    orders = []

    # If additional orders are provided, then use those!
    if additional_orders is None:
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order )
            Qmatrix = PolynomialParam.getJacobiEigenvectors(stackOfParameters[i])
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])


    else:
        for i in range(0, dimensions):
            orders.append(additional_orders[i])
            Qmatrix = PolynomialParam.getJacobiEigenvectors(stackOfParameters[i], orders[i])
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    # Compute multivariate Gauss points and weights
    p, w = getGaussianQuadrature(stackOfParameters, orders)

    # Evaluate the first point to get the size of the system
    fun_value_first_point = function(p[0,:])
    u0 =  q0[0,0] * fun_value_first_point
    N = 1
    gn = int(np.prod(orders))
    Uc = np.zeros((N, gn))
    Uc[0,1] = u0

    function_values = np.zeros((1,gn))
    for i in range(0, gn):
        function_values[0,i] = function(p[i,:])

    # Now we evaluate the solution at all the points
    for j in range(0, gn): # 0
        Uc[0,j]  = q0[0,j] * function_values[0,j]

    # Compute the corresponding tensor grid index set:
    order_correction = []
    for i in range(0, len(orders)):
        temp = orders[i] - 1
        order_correction.append(temp)

    tensor_grid_basis = IndexSet("tensor grid",  order_correction)
    tensor_set = IndexSet.getIndexSet(tensor_grid_basis)


    # Now we use kronmult
    K = efficient_kron_mult(Q, Uc)
    F = function_values

    return K, tensor_set

# Efficient kronecker product multiplication
# Adapted from David Gelich and Paul Constantine's kronmult.m
def efficient_kron_mult(Q, Uc):
    N = len(Q)
    n = np.zeros((N,1))
    nright = 1
    nleft = 1
    for i in range(0,N-1):
        rows_of_Q = len(Q[i])
        n[i,0] = rows_of_Q
        nleft = nleft * n[i,0]

    nleft = int(nleft)
    n[N-1,0] = len(Q[N-1]) # rows of Q[N]

    for i in range(N-1, -1, -1):
        base = 0
        jump = n[i,0] * nright
        for k in range(0, nleft):
            for j in range(0, nright):
                index1 = base + j
                index2 = int( base + j + nright * (n[i] - 1) )
                indices_required = np.arange(int( index1 ), int( index2 + 1 ), int( nright ) )
                small_Uc = np.mat(Uc[:, indices_required])
                temp = np.dot(Q[i] , small_Uc.T )
                temp_transpose = temp.T
                Uc[:, indices_required] = temp_transpose
            base = base + jump
        temp_val = np.max([i, 0]) - 1
        nleft = int(nleft/(1.0 * n[temp_val,0] ) )
        nright = int(nright * n[i,0])

    return Uc

# This function returns subsampled Gauss quadrature points and weights without ever
# computing the full tensor grid.
def getSubsampledGaussianQuadrature(self, subsampled_indices):

    stackOfParameters = self.uq_parameters
    indexSets = self.indexsets
    dimensions = len(stackOfParameters)
    multivariate_orders = np.zeros((1, dimensions))
    univariate_points_in_each_dimension = {}
    univariate_weights_in_each_dimension = {}

    # Final points & weights storage
    gauss_points = np.zeros((len(subsampled_indices), dimensions))
    gauss_weights = np.zeros((len(subsampled_indices)))

    # Total orders in each direction!
    for i in range(0, dimensions):
        p_local, w_local = PolynomialParam.getLocalQuadrature(stackOfParameters[i], [])
        univariate_points_in_each_dimension[i] = p_local
        univariate_weights_in_each_dimension[i] = w_local
        multivariate_orders[0,i] = stackOfParameters[i].order

    # Cycle through all the subsampled indices
    for i in range(0, len(subsampled_indices)):

        # From the large tensor grid, find the index set value of the "subsampled_index"
        index_set_entry = np.unravel_index(subsampled_indices[i], multivariate_orders[0])

        # Cycle through all the dimensions and determine corresponding quadrature points
        # and compute the corresponding weights
        weight_temp = 1.0
        for j in range(0, dimensions):
            individual_order = index_set_entry[j]
            gauss_points[i,j] = univariate_points_in_each_dimension[j][individual_order]
            weight = weight_temp * univariate_weights_in_each_dimension[j][individual_order]
            weight_temp = weight
        gauss_weights[i] = weight

    return gauss_points, gauss_weights

# Computes nD quadrature points and weights using a kronecker product
def getGaussianQuadrature(stackOfParameters, additional_orders=None):

    # Initialize some temporary variables
    dimensions = int(len(stackOfParameters))
    orders = []

    # Check for extra input argument!
    if additional_orders is None:
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order)
    else:
        for i in range(0, dimensions):
            orders.append(additional_orders[i])

    # Initialize points and weights
    pp = [1.0]
    ww = [1.0]

     # number of parameters
    # For loop across each dimension
    for u in range(0,dimensions):

        # Call to get local quadrature method (for dimension 'u')
        local_points, local_weights = PolynomialParam.getLocalQuadrature(stackOfParameters[u], orders[u])

        # Tensor product of the weights
        ww = np.kron(ww, local_weights)

        # Tensor product of the points
        dummy_vec = np.ones((len(local_points), 1))
        dummy_vec2 = np.ones((len(pp), 1))
        left_side = np.array(np.kron(pp, dummy_vec))
        right_side = np.array( np.kron(dummy_vec2, local_points) )
        pp = np.concatenate((left_side, right_side), axis = 1)

    # Ignore the first column of pp
    points = pp[:,1::]
    weights = ww

    # Now re-scale the points and return only if its not a Gaussian!
    for i in range(0, dimensions):
        for j in range(0, len(points)):
            if (stackOfParameters[i].param_type != "Gaussian" and stackOfParameters[i].param_type != "Normal" and stackOfParameters[i].param_type != "Beta")  :
                points[j,i] = 0.5 * ( points[j,i] + 1.0 )*( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound

            elif (stackOfParameters[i].param_type == "Beta" ):
                points[j,i] =  ( points[j,i] )*( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound

            # Scale points by the mean!
            elif (stackOfParameters[i].param_type == "Gaussian" or stackOfParameters[i].param_type == "Normal" ):
                points[j,i] = points[j,i] + float(stackOfParameters[i].shape_parameter_A)

    # Return tensor grid quad-points and weights
    return points, weights

# determines a multivariate orthogonal polynomial corresponding to the stackOfParameters,
# their corresponding orders and then evaluates the polynomial at the corresponding
# stackOfPoints.
def getMultiOrthoPoly(self, stackOfPoints):

    # "Unpack" parameters from "self"
    stackOfParameters = self.uq_parameters
    index_set = self.indexsets

    dimensions = len(stackOfParameters)
    p = {}

    # Save time by returning if univariate!
    if(dimensions == 1):
        poly , V =  PolynomialParam.getOrthoPoly(stackOfParameters[0], stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            p[i] = PolynomialParam.getOrthoPoly( stackOfParameters[i], stackOfPoints[:,i])

    # Now we multiply components according to the index set
    no_of_points = len(stackOfPoints)
    polynomial = np.zeros((len(index_set), no_of_points))
    for i in range(0, len(index_set)):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][int(index_set[i,k])] * temp
            temp = polynomial[i,:]

    return polynomial

# A method that returns all the indicies that have the same elements as the index_value
def find_repeated_elements(index_value, matrix):
    i = index_value
    selected_cell_indices = matrix[i,1::]
    local_store = [i]
    for j in range(0, len(matrix)):
        if(j != any(local_store) and i!=j): # to get rid of repeats because of two for loop structure
            if( all(matrix[j,1::] == selected_cell_indices)  ): # If all the indices match---i.e., the specific index is repeated
                local_store = np.append(local_store, [j])
    return local_store

# Method for getting a vector of function evaluations!
def evalfunction(function, points):
    function_values = np.zeros((1,len(points)))

    # For loop through all the points
    for i in range(0, len(points)):
        function_values[0,i] = function(points[i,:])

    return function_values

def error_function(string_value):
    print string_value
    sys.exit()
