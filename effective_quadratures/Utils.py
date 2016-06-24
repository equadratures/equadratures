#!/usr/bin/env python
import numpy as np
"""
    Set of utility functions that are used throughout EFFECTIVE-QUADRATURES
"""

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

def lineup(coefficients, index_set):
    orders_length = len(index_set[0])
    coefficient_array = np.zeros((len(coefficients), orders_length +1))
    for i in range(0, len(coefficients)):
        coefficient_array[i,0] = coefficients[i]
        for j in range(0, orders_length):
            coefficient_array[i,j+1] =  index_set[i,j]

    return coefficient_array
