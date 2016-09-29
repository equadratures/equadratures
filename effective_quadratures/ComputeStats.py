"""Utilities for exploiting active subspaces when optimizing."""
import numpy as np
from utils import error_function

class Statistics(object):
    
    """
    This subclass is an domains.ActiveVariableMap specifically for optimization.

    **See Also**

    optimizers.BoundedMinVariableMap
    optimizers.UnboundedMinVariableMap

    **Notes**

    This class's train function fits a global quadratic surrogate model to the
    n+2 active variables---two more than the dimension of the active subspace.
    This quadratic surrogate is used to map points in the space of active
    variables back to the simulation parameter space for minimization.
    """

    # constructor
    def __init__(self, coefficients, index_set):
        self.coefficients = coefficients
        self.index_set = index_set

    def getMean(self):
        """
        Train the global quadratic for the regularization.

        :param ndarray Y: N-by-n matrix of points in the space of active
            variables.
        :param int N: merely there satisfy the interface of `regularize_z`. It
            should not be anything other than 1.

        :return: Z, N-by-(m-n)-by-1 matrix that contains a value of the inactive
            variables for each value of the inactive variables.
        :rtype: ndarray

        **Notes**

        In contrast to the `regularize_z` in BoundedActiveVariableMap and
        UnboundedActiveVariableMap, this implementation of `regularize_z` uses
        a quadratic program to find a single value of the inactive variables
        for each value of the active variables.
        """        
        coefficients = self.coefficients
        mean = coefficients[0,0]
        return mean
    
    def getVariance(self):
        coefficients = self.coefficients
        m, n = coefficients.shape
        if m > n:
            coefficients = coefficients.T
        variance = np.sum(coefficients[0][1:m]**2)
        return variance

    # Function that computes first order Sobol' indices
    def getFirstOrderSobol(self):

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
