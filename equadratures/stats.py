"""Computing statistics from a polynomial expansions."""
from equadratures.basis import Basis
import numpy as np
from itertools import *

class Statistics(object):
    """
    Definition of a statistics object.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    :param numpy.ndarray coefficients: Coefficients from a polynomial expansion.
    :param numpy.ndarray quadrature_points: Quadrautre points associated with a quadrature rule of shape (number_of_points, dimensions)
    :param numpy.ndarray quadrature_weights: Quadrature weights associated with a quadrature rule of shape (number_of_points, 1)
    :param numpy.ndarray polynomial_matrix: The vandermonde-type matrix with multivariate polynomials evaluated at the quadrature_points of shape (cardinality, number_of_points).
    :param int max_sobol_order: For fast numerical calculations, max_sobol_order restricts the computation of conditional variances (and thus higher order Sobol' indices) to a
        user-defined order.

    """

    # constructor
    def __init__(self, parameters, basis, coefficients, quadrature_points=None, quadrature_weights=None, polynomial_matrix=None, max_sobol_order=None):
        mm = len(coefficients)
        self.coefficients = np.reshape(np.asarray(coefficients), (mm, 1))
        self.basis = basis
        self.parameters = parameters #should be a list containing instances of Parameter
        self.max_sobol_order = max_sobol_order
        self.__mean = private_get_mean(self.coefficients)
        self.__variance = private_get_variance(self.coefficients)
        self.__sobol = private_get_all_sobol_indices(self.coefficients, self.basis, max_sobol_order)
        # Only required when computing skewness and kurtosis.
        if (quadrature_points is None) and (quadrature_weights is None) and (polynomial_matrix is None):
            pass
        else:
            nn = len(quadrature_weights)
            weighted_evals = np.zeros((mm, nn))
            weighted_evals = polynomial_matrix * self.coefficients
            self.__weighted_evals = weighted_evals
            self.quadrature_weights = quadrature_weights
            self.__skewness = private_get_skewness(self.quadrature_weights, self.__weighted_evals, self.basis, self.__variance)
            self.__kurtosis = private_get_kurtosis(self.quadrature_weights, self.__weighted_evals, self.basis, self.__variance)
    def get_mean(self):
        """
        Compute the mean of the polynomial expansion.

        :param Statistics self:
            An instance of the Statistics class.

        :return:
            **mean**: The approximated mean of the polynomial fit; output as a float.
        """
        return self.__mean
    def get_variance(self):
        """
        Compute the variance of the polynomial expansion.

        :param Statistics self:
            An instance of the Statistics class.

        :return:
            **variance**: The approximated variance of the polynomial fit; output as a float.
        """
        return self.__variance
    def get_skewness(self):
        """
        Compute the skewness of the polynomial expansion.

        :param Statistics self:
            An instance of the Statistics class.

        :return:
            **skewness**: The approximated skewness of the polynomial fit; output as a float.
        """
        return self.__skewness
    def get_kurtosis(self):
        """
        Compute the kurtosis of the polynomial expansion.

        :param Statistics self:
            An instance of the Statistics class.

        :return:
            **kurtosis**: The approximated kurtosis of the polynomial fit; output as a float.
        """
        return self.__kurtosis
    def get_sobol(self, order=1):
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
        return {key: value for key, value in self.__sobol.items() if len(key) == order}
    def get_conditional_skewness(self, order=1):
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
        return private_conditional_skewness(order, self.quadrature_weights, \
            self.__weighted_evals, self.basis, self.__variance, self.__skewness)
    def get_conditional_kurtosis(self, order=1):
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
<<<<<<< HEAD
        return private_conditional_kurtosis(order, self.quadrature_weights, \
            self.__weighted_evals, self.basis, self.__variance, self.__kurtosis)
    def get_sobol_total(self):
=======
        return CondKurtosis(order, self.quad_wts, self.weighted_evals, self.basis, self.variance, self.kurtosis)

    def calc_TSI(self):
>>>>>>> c8e6b636b5b5a1f8c95bb385c61791a179f3ca21
        """
        Get total Sobol' indices
        :return: list: Totol Sobol' indices for each parameter
        """
<<<<<<< HEAD
        all_sobols = self.__sobol
=======
        all_sobols = self.sobol
>>>>>>> c8e6b636b5b5a1f8c95bb385c61791a179f3ca21
        dims = len(self.parameters)
        TSI = np.zeros(dims)
        for i in all_sobols.keys():
            for p in i:
                TSI[p] += all_sobols[i]
        return TSI
def private_get_mean(coefficients):
    return float(coefficients[0])
def private_get_variance(coefficients):
    result = 0.
    for i in range(1, len(coefficients)):
        variance = result + float(coefficients[i]**2)
        result = variance
    return variance
def private_get_all_sobol_indices(coefficients, basis, max_order):
    variance = private_get_variance(coefficients)
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
def private_get_skewness(quadrature_weights, weighted_evals, basis, variance):
    total_evals = np.sum(weighted_evals[1:],0)
    third_total_evals = total_evals**3
    return np.dot(third_total_evals,quadrature_weights)/(variance**1.5)
def private_get_kurtosis(quadrature_weights, weighted_evals, basis, variance):
    total_evals = np.sum(weighted_evals[1:],0)
    fourth_total_evals = total_evals**4
    return np.dot(fourth_total_evals,quadrature_weights)/(variance**2)
def private_conditional_skewness(order, quadrature_weights, weighted_evals, basis, variance, skewness):
    dimensions = basis.elements.shape[1]
    norm_ind = basis.elements.copy()
    norm_ind = list(map(tuple,(norm_ind > 0).astype(int)))
    combo_index = {}
    for i in combinations(range(dimensions), order):
        index = np.zeros(dimensions).astype(int)
        index[list(i)] = 1
        combo_index[tuple(index)] = 0.0
    #1st term
    cubed_evals = weighted_evals**3
    integral1 = np.dot(cubed_evals, quadrature_weights)
    for i in range(1, integral1.shape[0]):
        if sum(norm_ind[i]) == order:
            combo_index[norm_ind[i]] = combo_index[norm_ind[i]] + integral1[i] /(variance**1.5 * skewness)
    valid_indices = []
    for i in range(1, basis.cardinality):
        if sum(norm_ind[i]) <= order:
            valid_indices.append(i)
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
            evals2 = (weighted_evals[p,:]**2)*weighted_evals[q,:]
            integral2 = np.dot(evals2, quadrature_weights)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 3 * integral2 /(variance**1.5* skewness)
    temp_ind = basis.elements.copy()
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
                delta = False
                for d in range(dimensions):
                    if delta_pqr([temp_ind[p,:],temp_ind[q,:],temp_ind[r,:]]):
                        delta = True
                        break
                if delta:
                    continue

                evals3 = weighted_evals[p,:]*weighted_evals[q,:]*weighted_evals[r,:]

                integral3 = np.dot(evals3, quadrature_weights)

                combo_index[summed_norm_index] = combo_index[summed_norm_index] + 6 * integral3 /(variance**1.5* skewness)
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.items()}
    return combo_index
def private_conditional_kurtosis(order, quadrature_weights, weighted_evals, basis, variance, kurtosis):
    #Get all polynomials evaluated at the quad. pts and corresponding wts
    dimensions = basis.elements.shape[1]
    norm_ind = basis.elements.copy()
    norm_ind = list(map(tuple,(norm_ind > 0).astype(int)))
    combo_index = {}
    for i in combinations(range(dimensions), order):
        index = np.zeros(dimensions).astype(int)
        index[list(i)] = 1
        combo_index[tuple(index)] = 0.0
    #1st term
    fourth_evals = weighted_evals**4
    integral1 = np.dot(fourth_evals, quadrature_weights)
    for i in range(1, integral1.shape[0]):
        if sum(norm_ind[i]) == order:
            combo_index[norm_ind[i]] = combo_index[norm_ind[i]] + integral1[i] /(variance**2 * kurtosis)

    valid_indices = []
    for i in range(1, basis.cardinality):
        if sum(norm_ind[i]) <= order:
            valid_indices.append(i)
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
            integral2 = np.dot(evals2, quadrature_weights)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 4 * integral2 /(variance**2 * kurtosis)
    #3rd term (Can we avoid for loops in the future?)
    for a in range(len(valid_indices)):
        for b in range(a+1,len(valid_indices)):
            p = valid_indices[a]
            q = valid_indices[b]
            summed_norm_index = tuple(np.logical_or(norm_ind[p], norm_ind[q]).astype(int))
            if sum(summed_norm_index) != order:
                continue
            evals3 = (weighted_evals[p,:]**2)*(weighted_evals[q,:]**2)
            integral3 = np.dot(evals3, quadrature_weights)
            combo_index[summed_norm_index] = combo_index[summed_norm_index] + 6 * integral3 /(variance**2 * kurtosis)
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

                integral4 = np.dot(evals4, quadrature_weights)

                combo_index[summed_norm_index] = combo_index[summed_norm_index] + 12 * integral4 /(variance**2 * kurtosis)
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
                    integral5 = np.dot(evals5, quadrature_weights)
                    summed_norm_index = tuple(np.logical_or(np.logical_or(np.logical_or(norm_ind[p], norm_ind[q]), norm_ind[r]), norm_ind[t]).astype(int))
                    combo_index[summed_norm_index] = combo_index[summed_norm_index] + 24 * integral5 /(variance**2 * kurtosis)
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.items()}
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
