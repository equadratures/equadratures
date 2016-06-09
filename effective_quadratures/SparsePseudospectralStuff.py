#!/usr/bin/python
import PolyMethod as polmeth
import IndexSets as indi
import numpy as np
import Integration as integrals
import LeastSquares as lsqr

"""

this will be incorporated into PolyParentFile.py


"""
#
#------------------------------------------------------------------------------
def getSPAM_LSQRCoefficients(stackOfParameters, function, sparse_growth_rule, sparse_level):


    # 1. For a given tensor order
        # ---> compute tensor grid points and weights
        # ---> Use these to generate the A matrix and the b vector
        # ---> Compute x
    # 2. Pair the coefficients together with the index set using sparse "a" constant
    # 3. Now try to do this with Clenshaw-Curtis quadrature!!!

    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, sg_set_full = indi.sparse_grid_index_set(dimensions, sparse_level, sparse_growth_rule)
    rows = len(sparse_indices)

    individual_tensor_coefficients = {}
    indices = np.zeros((rows,1))
    for i in range(0, rows):
        index_set = indi.tensor_grid_index_set(sparse_indices[i,:])
        quad_pts, quad_weights = integrals.TensorGrid(stackOfParameters, sparse_indices[i,:])
        v = lsqr.solve(stackOfParameters, index_set, quad_pts, quad_weights, function)
        individual_tensor_coefficients[i] = v[0]
        indices[i,0] = len(individual_tensor_coefficients[i])

    sum_indices = int(np.sum(indices))
    store = np.zeros((sum_indices, dimensions+1))
    counter = int(0)

    for i in range(0,rows):
        for j in range(0, len(individual_tensor_coefficients[i])):
            store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][j]
            counter = counter + 1


    for i in range(0, len(sg_set_full)):
        for d in range(0, dimensions):
                 store[i,d+1] = sg_set_full[i,d]


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

    # For cleaning up!
    indices_to_delete = np.arange(counter, sum_indices, 1)
    final_store = np.delete(final_store, indices_to_delete, axis=0)

    return final_store

#------------------------------------------------------------------------------
#
# Regular sparse grid integration (not the right way!!!)
#
#------------------------------------------------------------------------------
def getSparseCoefficients(stackOfParameters, function, sparse_growth_rule, sparse_level):


    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, sg_set_full = indi.sparse_grid_index_set(dimensions, sparse_level, sparse_growth_rule)
    rows = len(sg_set_full)

    # Sparse grid integration rule
    pts, wts = integrals.SparseGrid(stackOfParameters, sparse_level, sparse_growth_rule)
    Wdiag = np.diag(wts)

    # Get multivariate orthogonal polynomial according to the index set
    P = polmeth.getMultiOrthoPoly(stackOfParameters, sg_set_full, pts)
    f = evalfunction(function, pts)
    f = np.mat(f)

    coefficients = np.zeros((rows, 1))
    for i in range(0,rows):
        coefficients[i,0] = np.mat(P[i,:]) * Wdiag * f.T

    return coefficients, sg_set_full



#------------------------------------------------------------------------------
#
# SPAM METHOD
#
#------------------------------------------------------------------------------
def getCoeficients(stackOfParameters, function, sparse_growth_rule, sparse_level):
    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, not_used = indi.sparse_grid_index_set(dimensions, sparse_level, sparse_growth_rule)
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
        orders = sparse_indices[i,:]
        K, I, F = polmeth.getPseudospectralCoefficients(stackOfParameters, orders, function)
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

    # For cleaning up!
    indices_to_delete = np.arange(counter, sum_indices, 1)
    final_store = np.delete(final_store, indices_to_delete, axis=0)

    return final_store

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
