#!/usr/bin/env python
import numpy as np
import effective_quadratures.Utils as util
"""

    Compute statistics and various sensitivity indices

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri

    To do:
    1. Code up Gianluca Geraci's skewness and kurtosis results
"""
def compute_mean_variance(coefficients, index_set):
    m, n = coefficients.shape
    if m > n:
        coefficients = coefficients.T

    mean = coefficients[0,0]
    variance = np.sum(coefficients[0,1::]**2)
    return mean, variance

# Function that computes first order Sobol' indices
def compute_first_order_Sobol_indices(coefficients, index_set):

    # Allocate memory!
    mean, variance = compute_mean_variance(coefficients, index_set)
    dimensions = len(index_set[0,:])


    if dimensions == 1:
        utils.error_function('ERROR: Sobol indices can only be computed for parameter studies with more than one parameter')
    else:
        index_set_entries = len(index_set[:,0])
        local_variance = np.zeros((index_set_entries, dimensions))
        first_order_sobol_indices = np.zeros((1, dimensions))

        # Loop for computing marginal variances!
        for j in range(0, dimensions):
            for i in range(0, index_set_entries): # no. of rows
                # If the index_set[0,j] is not zero but the remaining are...
                remaining_indices = np.arange(0, dimensions)
                remaining_indices = np.delete(remaining_indices, j)
                if(index_set[i,j] != 0 and np.sum(index_set[i, remaining_indices] )== 0):
                    local_variance[i, j] = coefficients[0,i]

        # Now take the sum of the squares of all the columns
        for j in range(0, dimensions):
            first_order_sobol_indices[0,j] = (np.sum(local_variance[:,j]**2))/(variance)

    return first_order_sobol_indices
