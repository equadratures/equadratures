#!/usr/bin/env python
import numpy as np
import sys
"""
    Set of utility functions that are used throughout EFFECTIVE-QUADRATURES
"""

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

# A simple utility that helps to visualize plotting sparse and tensor grid coefficients
def twoDgrid(coefficients, index_set):

    # First determine the maximum tensor grid order!
    max_order = int( np.max(index_set) ) + 1

    # Now create a tensor grid with this max. order
    y, x = np.mgrid[0:max_order, 0:max_order]
    z = (x*0 + y*0) + float('NaN')

    # Now for each grid point, cycle through spam_coefficients and see if
    # that grid point is present, if so, add the coefficient value to z.
    for i in range(0, max_order):
        for j in range(0, max_order):
            x_entry = x[i,j]
            y_entry = y[i,j]
            for k in range(0, len(index_set)):
                if(x_entry == index_set[k,0] and y_entry == index_set[k,1]):
                    z[i,j] = coefficients[0,k]

    return x,y,z, max_order


def compute_b_vector(quad_pts, function, quad_weights):
    f = np.mat( evalfunction(quad_pts, function) )
    W = np.diag( quad_weights )
    return W * f.T

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
