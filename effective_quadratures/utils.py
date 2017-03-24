#!/usr/bin/env python
"""Utilities for plotting and catching errors"""
import numpy as np
import sys

def column(matrix, i):
    return [row[i] for row in matrix]
    
# A sample utility to get a 2D meshgrid of points
def meshgrid(lower_lim, upper_lim, nx1, nx2):

    total_points = nx1 * nx2 # total points required!
    x1_pts = np.linspace(lower_lim, upper_lim, nx1)
    x2_pts = np.linspace(lower_lim, upper_lim, nx2)

    # Code segment below is solely for resizing *(must be a better way to do this!)
    x1, x2 = np.meshgrid(x1_pts, x2_pts, indexing='ij') # combined grid
    x1o = np.reshape(x1, (total_points, 1))
    x2o = np.reshape(x2, (total_points, 1))
    stackOfPoints = np.concatenate((x1o, x2o), axis = 1)

    return stackOfPoints, x1, x2



def compute_b_vector(quad_pts, function, quad_weights):
    f = np.mat( evalfunction(quad_pts, function) )
    W = np.diag( quad_weights )
    return W * f.T

# Evaluate the gradient of the function at given points
def evalgradients(points, fungrad, format):
    dimensions = len(points[0,:])

    if format is 'matrix':
        grad_values = np.zeros((len(points), dimensions))
        # For loop through all the points
        for i in range(0, len(points)):
            output_from_gradient_call = fungrad(points[i,:])
            for j in range(0, dimensions):
                grad_values[i,j] = output_from_gradient_call[j]
        return grad_values
    elif format is 'vector':
        grad_values = np.zeros((len(points) * dimensions, 1))
        # For loop through all the points
        counter = 0
        for i in range(0, len(points)):
            output_from_gradient_call = fungrad(points[i,:])
            for j in range(0, dimensions):
                grad_values[counter, 0] = output_from_gradient_call[j]
                counter = counter + 1
        return np.mat(grad_values)
    else:
        error_function('evalgradients(): Format must be either matrix or vector!')
        return 0

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

# Loop through the indices of the small index set, and find the corresponding coefficient value in the large one!
def compute_errors(coefficients_large, index_set_large, coefficients_small, index_set_small):
    no_of_small_coefficients  = len(coefficients_small)
    no_of_large_coefficients = len(coefficients_large)
    dimensions = len(index_set_small[0])
    error = np.ones((no_of_small_coefficients))
    counter = 0 
    for i in range(0, no_of_small_coefficients):    
        current_index  = index_set_small[i,:]
        for j in range(0, no_of_large_coefficients):
            temp_index = index_set_large[j, :]
            if compare_lists(current_index, temp_index):
                error[counter] = coefficients_large[j] - coefficients_small[i] 
                counter = counter + 1
                break

    return np.linalg.norm(error, 2), error
    # i suppose its worth checking to see that counter = no_of_small_coefficients

# Checks to make sure that two lists are identical (including ordering!)
def compare_lists(list1, list2):
    elements_in_list = len(list1)
    counter = 0
    for i in range(0, len(list1)):
        if list1[i] == list2[i]: 
            counter = counter + 1
    if counter == elements_in_list:
        return True
    else:
        return False


def lineup(coefficients, index_set):
    orders_length = len(index_set[0])
    coefficient_array = np.zeros((len(coefficients), orders_length +1))
    for i in range(0, len(coefficients)):
        coefficient_array[i,0] = coefficients[i]
        for j in range(0, orders_length):
            coefficient_array[i,j+1] =  index_set[i,j]

    return coefficient_array

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

# An error function
def error_function(string_value):
    print string_value
    sys.exit()
