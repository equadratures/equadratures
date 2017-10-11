"""Computing Statistics from Polynomial Expansions"""
# TODO: change the docs!
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
        * **self.sobol**:(dict) Sobol' indices of order up to number of dimensions.

    **Notes:** 
    In a future release we will be incorporating second order Sobol indices, skewness and kurtosis based indices. Stay tuned!
    """

    # constructor
    def __init__(self, coefficients, index_set):
        self.coefficients = coefficients
        self.index_set = index_set
        self.mean = getMean(self.coefficients)
        self.variance = getVariance(self.coefficients)
        self.sobol = getAllSobol(self.coefficients, self.index_set)
    
        
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
    
    def getSobol(self, order = 1):
        """
        Get Sobol' indices at specified order. 

        :param order int: The order at which Sobol' indices are computed. By default, computes first order Sobol' indices.
        :return: indices, Dictionary where keys specify non-zero dimensions and values represent Sobol' indices.
        :rtype: dict

        **Sample usage:**
        stats = Statistics(coeffcients, index_set)
        fosi = stats.getSobol(1)        
        
        """
        coefficients = self.coefficients
        index_set = self.index_set
        m, n = coefficients.shape
        variance = getVariance(coefficients)
        if m > n:
            coefficients = coefficients.T
        index_set = index_set.elements
        m, dimensions = index_set.shape
        assert(0 < order <= dimensions)        
        
        if dimensions == 1:
            return 1.0
        else:
            index_set_entries = m
    
            #Build dict to contain the Sobol' indices
            #Dict is a better idea than lists because ordering is not clear for higher order indices
            indices = {}
            
            for i in combinations(range(dimensions),order):
                indices[i] = 0
                
                
            for i in range(0,index_set_entries): #loop over rows
                row = index_set[i,:]
                non_zero_entries = np.nonzero(row)[0]
                non_zero_entries.sort()    #just in case
                if len(non_zero_entries) == order and coefficients[0][i]**2 > 1e-5: #neglect entries that should actually be zero
                    indices[tuple(non_zero_entries)] = indices[tuple(non_zero_entries)] + coefficients[0][i]**2 / variance
            
            return indices

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



# Function that computes the Sobol' indices of all orders up to dimension of i/p
def getAllSobol(coefficients, index_set):
    m, n = coefficients.shape
    variance = getVariance(coefficients)
    if m > n:
        coefficients = coefficients.T
    index_set = index_set.elements
    m, dimensions = index_set.shape # m was simply coefficients.shape before. Review?

    if dimensions == 1:
        return {0:1.0}
    else:
        index_set_entries = m

        #Build dict to contain the Sobol' indices
        #Keys contain the non-zero indices
        #Values contain the Sobol' indices
                
        combo_index = {}
        
        for order in range(1,dimensions+1): #loop over order        
            for i in combinations(range(dimensions),order):
                #initialize each index to be 0                
                combo_index[i] = 0
                
            for i in range(0,index_set_entries): #loop over rows
                row = index_set[i,:]
                non_zero_entries = np.nonzero(row)[0]
                non_zero_entries.sort()    #just in case
                if len(non_zero_entries) == order and coefficients[0][i]**2 > 1e-5: #neglect entries that should actually be zero (what constitutes as zero?)
                    combo_index[tuple(non_zero_entries)] = combo_index[tuple(non_zero_entries)] + coefficients[0][i]**2 / variance
        
        check_sum = sum(combo_index.values())
        assert(abs(check_sum - 1.0) < 1e-5) 
        
        return combo_index