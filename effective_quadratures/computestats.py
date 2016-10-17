#!/usr/bin/env python
"""Computing Statistics from Polynomial Expansions"""
import numpy as np
from utils import error_function

class Statistics(object):
    """
    This class defines a Statistics object

    :param numpy matrix coefficients: polynomial coefficients (can be from any of the methods)
    :param IndexSet index_set: The index set corresponding to the polynomial basis that was used to compute the coefficients

    """

    # constructor
    def __init__(self, coefficients, index_set):
        self.coefficients = coefficients
        self.index_set = index_set

    def getMean(self):
        """
        Computes the mean of a polynomial expansion using its coefficients.

        :param Statistics self: An instance of the Statistics class
        :return: mean
        :rtype: double

        **Notes**
        The mean is simply the first coefficient of the expansion.
        """        
        coefficients = self.coefficients
        mean = coefficients[0,0]
        return mean
    
    def getVariance(self):
        """
        Computes the variance of a polynomial expansion using its coefficients.

        :param Statistics self: An instance of the Statistics class
        :return: variance
        :rtype: double

        **Notes**
        The variance is the sum of the squares of all the coefficients except the first coefficient.
        """  
        coefficients = self.coefficients
        m, n = coefficients.shape
        if m > n:
            coefficients = coefficients.T
        variance = np.sum(coefficients[0][1:m]**2)
        return variance

    # Function that computes first order Sobol' indices
    def getFirstOrderSobol(self):
        """
        Computes the first order Sobol indices.

        :param Statistics self: An instance of the Statistics class
        :return: first_order_sobol_indices
        :rtype: numpy ndarray
        """ 
        coefficients = self.coefficients
        m, n = coefficients.shape
        if m > n:
            coefficients = coefficients.T

        index_set = self.index_set

        # Allocate memory!
        index_set = index_set.getIndexSet()
        index_set = np.mat(index_set)
        m, dimensions =  index_set.shape
        variance = self.getVariance()

        if dimensions == 1:
            utils.error_function('ERROR: Sobol indices can only be computed for parameter studies with more than one parameter')
        else:
            index_set_entries = m
            local_variance = np.zeros((index_set_entries, dimensions))
            first_order_sobol_indices = np.zeros((dimensions))

            # Loop for computing marginal variances!
            for j in range(0, dimensions):
                for i in range(0, index_set_entries): # no. of rows
                    # If the index_set[0,j] is not zero but the remaining are...
                    remaining_indices = np.arange(0, dimensions)
                    remaining_indices = np.delete(remaining_indices, j)
                    if(index_set[i,j] != 0 and np.sum(index_set[i, remaining_indices] ) == 0):
                        local_variance[i, j] = coefficients[0][i]

            # Now take the sum of the squares of all the columns
            for j in range(0, dimensions):
                first_order_sobol_indices[j] = (np.sum(local_variance[:,j]**2))/(variance)

        return first_order_sobol_indices
