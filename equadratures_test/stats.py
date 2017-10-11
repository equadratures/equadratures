"""Computing Statistics from Polynomial Expansions"""
import numpy as np
from .plotting import barplot
from itertools import *
class Statistics(object):
    """
    :param numpy-matrix coefficients: Coefficients from a polynomial expansion. Can be computed using any technique.
    :param IndexSet index_set: Polynomial index set. If an index set is not given, the constructor uses a tensor grid basis of polynomials. For total order and hyperbolic index sets, the user needs to explicity input an index set.

    Attributes:
        * **self.mean**: (double) Mean of the polynomial expansion.
        * **self.variance**: (double) Variance of the polynomial expansion.
        * **self.fosi**:(numpy array) First order Sobol indices.

    **Notes:** 
    In a future release we will be incorporating second order Sobol indices, skewness and kurtosis based indices. Stay tuned!
    """

    # constructor
    def __init__(self, coefficients, index_set):
        self.coefficients = coefficients
        self.index_set = index_set
        self.mean = getMean(self.coefficients)
        self.variance = getVariance(self.coefficients)
        self.fosi = getfosi(self.coefficients, self.index_set)
        self.sobol = getSobol(self.coefficients, self.index_set)
    
        
    def plot(self, filename=None):
        """
        Produces a bar graph of the first order Sobol indices

        :param Statistics object: An instance of the Statistics class.
        :param string filename: A file name in case the user wishes to save the bar graph. The default output is an eps file.

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        # A bar graph plot of the Sobol indices!
        dimensions = self.index_set.dimension
        xbins = range(0, dimensions)
        barplot(xbins, self.fosi, 'Parameters', 'Sobol indices')


# Private functions!
def getMean(coefficients):
    mean = coefficients[0,0]
    return mean
        
def getVariance(coefficients):
    m, n = coefficients.shape
    if m > n:
        coefficients = coefficients.T
    variance = np.sum(coefficients[0][1:m]**2)
    return variance

# Function that computes first order Sobol' indices
def getfosi(coefficients, index_set):
    #change to compute only a specific order of indices
    m, n = coefficients.shape
    variance = getVariance(coefficients)
    if m > n:
        coefficients = coefficients.T
    index_set = index_set.elements
    m, dimensions = index_set.shape

    if dimensions == 1:
        return 1.0
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

# Function that computes the Sobol' indices of given order
def getSobol(coefficients, index_set):
    m, n = coefficients.shape
    variance = getVariance(coefficients)
    if m > n:
        coefficients = coefficients.T
    index_set = index_set.elements
    print(coefficients.T.shape)
    print(index_set.shape)
    combined = np.concatenate((index_set, coefficients.T),1)
    combined = combined[combined[:,3] > 0.1]
    print(combined.astype(int))
    m, dimensions = index_set.shape

    if dimensions == 1:
        return 1.0
    else:
        index_set_entries = m
#        local_variance = np.zeros((index_set_entries, dimensions))
#        first_order_sobol_indices = np.zeros((dimensions))
        # Loop for computing marginal variances!
#        for j in range(0, dimensions):
#            for i in range(0, index_set_entries): # no. of rows
#                # If the index_set[0,j] is not zero but the remaining are...
#                remaining_indices = np.arange(0, dimensions)
#                remaining_indices = np.delete(remaining_indices, j)
#                if(index_set[i,j] != 0 and np.sum(index_set[i, remaining_indices] ) == 0):
#                    local_variance[i, j] = coefficients[0][i]
        
        #Build dict to contain the Sobol' indices
                
        combo_index = {}
        
        for order in range(1,dimensions+1): #loop over order        
            for i in combinations(range(dimensions),order):
                combo_index[i] = 0
                
                
            for i in range(0,index_set_entries): #loop over rows
                row = index_set[i,:]
    #            print(row)
                non_zero_entries = np.nonzero(row)[0]
                non_zero_entries.sort()    #just in case
    #            print(non_zero_entries)
                if len(non_zero_entries) == order and coefficients[0][i]**2 > 1e-5: #neglect entries that should actually be zero
                    combo_index[tuple(non_zero_entries)] = combo_index[tuple(non_zero_entries)] + coefficients[0][i]**2 / variance
                
        
        
#        print(combo_index)
            
         # Now take the sum of the squares of all the columns
#        for j in range(0, dimensions):
#            first_order_sobol_indices[j] = (np.sum(local_variance[:,j]**2))/(variance)
        
        return combo_index