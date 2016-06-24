#!/usr/bin/env python
import numpy as np
"""

Set of functions that compute statistics based on the coefficient values and their index set
We assume coefficients and index sets are correctly computed.

1. mean
2. variance
3. sobol indices - first order and higher
4. skewness
5. kurtosis

"""
def compute_mean_variance(coefficients, index_set):
    mean = coefficients[0,0]
    variance = np.sum(coefficients[0,1::]**2)
    return mean, variance


def compute_first_order_Sobol_indices(coefficients, index_set):


    return 0
