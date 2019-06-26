"""Computing statistics from polynomial expansions."""
import numpy as np
from .basis import Basis
from itertools import *

class Statistics(object):
    """
    :param numpy-matrix coefficients: Coefficients from a polynomial expansion. Can be computed using any technique.
    :param Basis basis: Polynomial index set. If an index set is not given, the constructor uses a tensor grid basis of polynomials. For total order and hyperbolic index sets, the user needs to explicity input an index set.
    Attributes:
        * **self.mean**: (double) Mean of the polynomial expansion.
        * **self.variance**: (double) Variance of the polynomial expansion.
        * **self.sobol**:(dict) Sobol' indices of order up to number of dimensions.
    """

    # constructor
    def __init__(self, coefficients, basis, parameters,  quadrature_points=None, quadrature_weights=None, polynomial_evals=None,max_sobol_order = None,):
        mm = len(coefficients)
        self.coefficients = np.reshape(np.asarray(coefficients), (mm, 1))
        self.basis = basis
        self.parameters = parameters #should be a list containing instances of Parameter

        self.mean = getMean(self.coefficients)
        self.variance = getVariance(self.coefficients)
        self.sobol = getAllSobol(self.coefficients, self.basis, max_sobol_order)

        #Prepare evals of polynomials for skewness and kurtosis
        if (quadrature_points is None) and (quadrature_weights is None) and (polynomial_evals is None):
            pass
        else:
            nn = len(quadrature_weights)
            weighted_evals = np.zeros((mm, nn))
            weighted_evals = polynomial_evals * self.coefficients
            self.weighted_evals = weighted_evals
            self.quad_wts = quadrature_weights
            self.skewness = getSkewness(self.quad_wts, self.weighted_evals, self.basis, self.variance)
            self.kurtosis = getKurtosis(self.quad_wts, self.weighted_evals, self.basis, self.variance)

    def getSobol(self, order = 1):
        """
        Get Sobol' indices at specified order.

        :param order int: The order at which Sobol' indices are computed. By default, computes first order Sobol' indices.

        :return: indices: Dictionary where keys specify non-zero dimensions and values represent Sobol' indices.
        :rtype: dict

        **Sample usage:**

        .. code-block:: python

            stats = Statistics(coeffcients, basis)
            fosi = stats.getSobol(1)

        """
        return {key: value for key, value in self.sobol.items() if len(key) == order}


    def getCondSkewness(self, order = 1):
        """
        Get conditional skewness indices at specified order.

        :param order int: The order at which conditional skewness indices are computed. By default, computes first order conditional skewness.

        :return: indices: Dictionary where keys specify non-zero dimensions and values represent conditional skewness indices.
        :rtype: dict

        **Sample usage:**

        .. code-block:: python

            stats = Statistics(coeffcients, basis)
            first_order_skewness = stats.getCondSkewness(1)

        """
        return CondSkewness(order, self.quad_wts, self.weighted_evals, self.basis, self.variance, self.skewness)
    def getCondKurtosis(self, order = 1):
        """
        Get conditional kurtosis indices at specified order.

        :param order int: The order at which conditional kurtosis indices are computed. By default, computes first order conditional kurtosis.

        :return: indices: Dictionary where keys specify non-zero dimensions and values represent conditional kurtosis indices.
        :rtype: dict

        **Sample usage:**

        .. code-block:: python

            stats = Statistics(coeffcients, basis)
            first_order_kurtosis = stats.getCondKurtosis(1)

        """
        return CondKurtosis(order, self.quad_wts, self.weighted_evals, self.basis, self.variance, self.kurtosis)

    #Calculates the total sensitivity based on list of input dicts
    #Assumes they are ordered so that the first element is the first order indices!
    @staticmethod
    def calc_TSI(list_of_indices_dicts):
        dim = len(list_of_indices_dicts[0].keys())
        TSI = np.zeros((dim))
        for i in range(len(list_of_indices_dicts)):
            for j in range(dim):
                for k in list_of_indices_dicts[i].keys():
                    if j in k:
                        TSI[j] = TSI[j] + list_of_indices_dicts[i][k]

        return TSI

# Private functions!
def getMean(coefficients):
    return float(coefficients[0])

def getVariance(coefficients):
    result = 0.
    for i in range(1, len(coefficients)):
        variance = result + float(coefficients[i]**2)
        result = variance
    return variance

# Function that computes the Sobol' indices of all orders up to dimension of i/p
def getAllSobol(coefficients, basis, max_order):
    variance = getVariance(coefficients)
    if not(isinstance(basis, np.ndarray)):
        basis = basis.elements
    m, dimensions = basis.shape
    if dimensions == 1:
        return {0:1.0}
    else:
        basis_entries = m
        combo_index = {}
        if max_order is None or max_order > dimensions:
            max_order = dimensions
        for order in range(1,max_order+1): #loop over order
            for i in combinations(range(dimensions),order):
                #initialize each index to be 0
                combo_index[i] = 0


            for i in range(0,basis_entries): #loop over rows
                row = basis[i,:]
                non_zero_entries = np.nonzero(row)[0]
                non_zero_entries.sort()    #just in case
                if len(non_zero_entries) == order: #neglect entries that should actually be zero (what constitutes as zero?)
                    combo_index[tuple(non_zero_entries)] = float(combo_index[tuple(non_zero_entries)] + coefficients[i]**2 / variance)
        check_sum = sum(combo_index.values())
        if (abs(check_sum - 1.0) >= 1e-2):
            print("Possible discrepancy in calculation, sum of indices = " + str(check_sum))

        return combo_index

# Return global skewness
def getSkewness(quad_wts, weighted_evals, basis, variance):
    total_evals = np.sum(weighted_evals[1:],0)
#    print weighted_evals[0]
    third_total_evals = total_evals**3
#    print third_total_evals.shape
#    print quad_wts.shape

    return np.dot(third_total_evals,quad_wts)/(variance**1.5)

# Return global kurtosis
def getKurtosis(quad_wts, weighted_evals, basis, variance):
    total_evals = np.sum(weighted_evals[1:],0)
    fourth_total_evals = total_evals**4

    return np.dot(fourth_total_evals,quad_wts)/(variance**2)

# Return conditional skewness of specified order, in dictionary format similar to Sobol' indices
#Unfortunately, to compute conditional indices, this slow method must be used!
def CondSkewness(order, quad_wts, weighted_evals, basis, variance, skewness):
    #Get all polynomials evaluated at the quad. pts and corresponding wts

    dimensions = basis.elements.shape[1]
    norm_ind = basis.elements.copy()
    norm_ind = list(map(tuple,(norm_ind > 0).astype(int)))

    combo_index = {}
#    for tot_order in range(1,dimensions+1): #loop over order
    for i in combinations(range(dimensions), order):
        #initialize each index of the specified order to be 0
#        if sum(i) != order:
#            continue
#        combo_index[i] = 0.0
        index = np.zeros(dimensions).astype(int)
        index[list(i)] = 1
        combo_index[tuple(index)] = 0.0

    #1st term
    cubed_evals = weighted_evals**3
    integral1 = np.dot(cubed_evals, quad_wts)
    for i in range(1, integral1.shape[0]):
        if sum(norm_ind[i]) == order:
            combo_index[norm_ind[i]] = combo_index[norm_ind[i]] + integral1[i] /(variance**1.5 * skewness)

#    print combo_index
    valid_indices = []
    for i in range(1, basis.cardinality):
        if sum(norm_ind[i]) <= order:
            valid_indices.append(i)
#    print valid_indices
    #2nd term (Can we avoid for loops in the future?)
    for p in range(1,basis.cardinality):
        for q in range(1,basis.cardinality):
            summed_norm_index =tuple(np.logical_or(norm_ind[p], norm_ind[q]).astype(int))
            if sum(summed_norm_index) != order:
                continue
            #check if selection function is zero, in which case delta = True
            delta = False
            for d in range(dimensions):
                if (basis.elements[p,d] == 0 and basis.elements[q,d] != 0) or p == q:
                    delta = True
                    break
            if delta:
                continue
#            print basis.elements[p]
#            print basis.elements[q]
            evals2 = (weighted_evals[p,:]**2)*weighted_evals[q,:]
            integral2 = np.dot(evals2, quad_wts)
#            print basis.elements[p]
#            print basis.elements[q]
#            print 3* integral2 /(variance**1.5* skewness)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 3 * integral2 /(variance**1.5* skewness)

#    print combo_index
    temp_ind = basis.elements.copy()
    #3rd term (Can we avoid for loops in the future?)
    i = 0
    for a in range(len(valid_indices)):
        for b in range(a+1, len(valid_indices)):
            for c in range(b+1, len(valid_indices)):
                p = valid_indices[a]
                q = valid_indices[b]
                r = valid_indices[c]

                summed_norm_index =tuple(np.logical_or(np.logical_or(norm_ind[p], norm_ind[q]), norm_ind[r]).astype(int))
                if sum(summed_norm_index) != order:
                    continue
                #check if selection function is zero, in which case delta = True
#                print [p,q,r]
                delta = False
                for d in range(dimensions):
                    if delta_pqr([temp_ind[p,:],temp_ind[q,:],temp_ind[r,:]]):
                        delta = True
#                        print "hi"
                        break
                if delta:
                    continue

                evals3 = weighted_evals[p,:]*weighted_evals[q,:]*weighted_evals[r,:]

                integral3 = np.dot(evals3, quad_wts)

                combo_index[summed_norm_index] = combo_index[summed_norm_index] + 6 * integral3 /(variance**1.5* skewness)

#    print combo_index
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.items()}
    return combo_index

# Return conditional kurtosis of specified order, in dictionary format similar to Sobol' indices
def CondKurtosis(order, quad_wts, weighted_evals, basis, variance, kurtosis):
    #Get all polynomials evaluated at the quad. pts and corresponding wts
    dimensions = basis.elements.shape[1]
    norm_ind = basis.elements.copy()
    norm_ind = list(map(tuple,(norm_ind > 0).astype(int)))

    combo_index = {}
#    for tot_order in range(1,dimensions+1): #loop over order
    for i in combinations(range(dimensions), order):
        #initialize each index to be 0
#            if sum(i) != order:
#                continue
#            combo_index[i] = 0.0
        index = np.zeros(dimensions).astype(int)
        index[list(i)] = 1
        combo_index[tuple(index)] = 0.0
    #1st term
    fourth_evals = weighted_evals**4
    integral1 = np.dot(fourth_evals, quad_wts)
    for i in range(1, integral1.shape[0]):
        if sum(norm_ind[i]) == order:
            combo_index[norm_ind[i]] = combo_index[norm_ind[i]] + integral1[i] /(variance**2 * kurtosis)

    valid_indices = []
    for i in range(1, basis.cardinality):
        if sum(norm_ind[i]) <= order:
            valid_indices.append(i)
#    print combo_index
    #2nd term (Can we avoid for loops in the future?)
    for p in valid_indices:
        for q in valid_indices:
            summed_norm_index =tuple(np.logical_or(norm_ind[p], norm_ind[q]).astype(int))
            if sum(summed_norm_index) != order:
                continue
            #check if selection function is zero, in which case delta = True
            delta = False
            for d in range(dimensions):
                if (basis.elements[p,d] == 0 and basis.elements[q,d] != 0) or p == q:
                    delta = True
                    break
            if delta:
                continue
            evals2 = (weighted_evals[p,:]**3)*weighted_evals[q,:]
            integral2 = np.dot(evals2, quad_wts)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 4 * integral2 /(variance**2 * kurtosis)
#    print combo_index
    #3rd term (Can we avoid for loops in the future?)
    for a in range(len(valid_indices)):
        for b in range(a+1,len(valid_indices)):
            p = valid_indices[a]
            q = valid_indices[b]
            summed_norm_index = tuple(np.logical_or(norm_ind[p], norm_ind[q]).astype(int))
            if sum(summed_norm_index) != order:
                continue
            evals3 = (weighted_evals[p,:]**2)*(weighted_evals[q,:]**2)
            integral3 = np.dot(evals3, quad_wts)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 6 * integral3 /(variance**2 * kurtosis)

#    print combo_index
    #4th term (Can we avoid for loops in the future?)
    for a in range(len(valid_indices)):
        for b in range(len(valid_indices)):
            for c in range(b+1, len(valid_indices)):
                p = valid_indices[a]
                q = valid_indices[b]
                r = valid_indices[c]
                summed_norm_index = tuple(np.logical_or(np.logical_or(norm_ind[p], norm_ind[q]), norm_ind[r]).astype(int))
                if sum(summed_norm_index) != order:
                    continue
                #check if selection function is zero, in which case delta = True
                delta = False
                for d in range(dimensions):
                    if (delta_p_qr([basis.elements[p,:], basis.elements[q,:], basis.elements[r,:]]))\
                    or (q==p) or (r==p):
                        delta = True

                        break
                if delta:
                    continue
                evals4 = (weighted_evals[p,:]**2)*weighted_evals[q,:]*weighted_evals[r,:]

                integral4 = np.dot(evals4, quad_wts)

                combo_index[summed_norm_index] = combo_index[summed_norm_index] + 12 * integral4 /(variance**2 * kurtosis)
#    print combo_index
    #5th term (Can we avoid for loops in the future?) (especially this. Scales poorly!)
    temp_ind = basis.elements.copy()
    for a in range(len(valid_indices)):
        for b in range(a+1, len(valid_indices)):
            for c in range(b+1, len(valid_indices)):
                for d in range(c+1, len(valid_indices)):
                    p = valid_indices[a]
                    q = valid_indices[b]
                    r = valid_indices[c]
                    t = valid_indices[d]
                    summed_norm_index = tuple(np.logical_or(np.logical_or(np.logical_or(norm_ind[p], norm_ind[q]), norm_ind[r]), norm_ind[t]).astype(int))
                    if sum(summed_norm_index) != order:
                        continue
                    #check if selection function is zero, in which case delta = True
                    delta = False
                    for d in range(dimensions):
                        if delta_pqrs([temp_ind[p,:], temp_ind[q,:], temp_ind[r,:], temp_ind[t,:]]):
                            delta = True

                            break
                    if delta:
                        continue
                    evals5 = weighted_evals[p,:]*weighted_evals[q,:]*weighted_evals[r,:]*weighted_evals[t,:]
                    integral5 = np.dot(evals5, quad_wts)
                    summed_norm_index = tuple(np.logical_or(np.logical_or(np.logical_or(norm_ind[p], norm_ind[q]), norm_ind[r]), norm_ind[t]).astype(int))
                    combo_index[summed_norm_index] = combo_index[summed_norm_index] + 24 * integral5 /(variance**2 * kurtosis)
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.items()}
#    print combo_index
    return combo_index


#Calculates delta^p_{qr} (Geraci)
def delta_p_qr(rows):
    #first row = p, then q, then r
    assert(len(rows) == 3)
    dimensions = len(rows[0])
    for d in range(dimensions):
        if rows[0][d] == 0:
            if rows[1][d] != rows[2][d]:
                return True
    return False

# Calculates delta_pqrs (From Geraci)
def delta_pqrs(rows):
    assert(len(rows) == 4) #Comparison of 4 rows
    norm_rows = []
    norm_rows[:] = rows[:]
    dimensions = len(rows[0])
    for i in range(4):
        for j in range(dimensions):
            if rows[i][j] != 0:
                norm_rows[i][j] = 1
    for d in range(dimensions):
        col_sum = norm_rows[0][d] + norm_rows[1][d] + norm_rows[2][d] + norm_rows[3][d]
        if  col_sum == 1:
            return True
        elif col_sum == 2:
            indices = np.zeros((4))
            for k in range(4):
                indices[k] = rows[k][d]
            non_zeros = indices.nonzero()[0]
            assert(len(non_zeros) == 2)
            if indices[non_zeros][0] == indices[non_zeros][1]:
                pass
            else:
                return True
    return False

#Calculates delta_pqr (Geraci)
def delta_pqr(rows):
#    print "-----"
#    print rows
    assert(len(rows) == 3)
    norm_rows = []
    norm_rows[:] = rows[:]
    dimensions = len(rows[0])
    for i in range(3):
        for j in range(dimensions):
            if rows[i][j] != 0:
                norm_rows[i][j] = 1
#    print norm_rows
    for d in range(dimensions):
        col_sum = norm_rows[0][d] + norm_rows[1][d] + norm_rows[2][d]
#        print "dimesnion = " + str(d)
#        print col_sum

        if col_sum == 1:
            return True
        elif col_sum == 2:
            if (rows[0][d] == rows[1][d]) or (rows[1][d] == rows[2][d]) or (rows[0][d] == rows[2][d]): #Hack
                pass
            else:
                return True
    return False
