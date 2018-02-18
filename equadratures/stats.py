"""Computing Statistics from Polynomial Expansions"""
import numpy as np
from .plotting import barplot, triplebarplot, piechart, scatterplot2
from basis import Basis
from itertools import *
class Statistics(object):
    """
    :param numpy-matrix coefficients: Coefficients from a polynomial expansion. Can be computed using any technique.
    :param Basis basis: Polynomial index set. If an index set is not given, the constructor uses a tensor grid basis of polynomials. For total order and hyperbolic index sets, the user needs to explicity input an index set.

    Attributes:
        * **self.mean**: (double) Mean of the polynomial expansion.
        * **self.variance**: (double) Variance of the polynomial expansion.
        * **self.sobol**:(dict) Sobol' indices of order up to number of dimensions.

    **Notes:** 
    In a future release we will be incorporating second order Sobol indices, skewness and kurtosis based indices. Stay tuned!
    """

    # constructor
    def __init__(self, coefficients, basis, parameters, quadrature_points=None, quadrature_weights=None, polynomial_evals=None):
        self.coefficients = coefficients
        self.basis = basis
        self.parameters = parameters #should be a list containing instances of Parameter
        
        self.mean = getMean(self.coefficients)
        self.variance = getVariance(self.coefficients)
        self.sobol = getAllSobol(self.coefficients, self.basis)
        
        #Prepare evals of polynomials for skewness and kurtosis
        if (quadrature_points is None) and (quadrature_weights is None) and (polynomial_evals is None):
            # now stats embedded in polyreg no need for this.
#            polyreg = Polyreg(parameters, basis)
#            quad_pts, quad_wts = polyreg.getQuadratureRule()
#            evals,deriv = polyreg.getPolynomial_t(quad_pts)
#            self.weighted_evals = evals * coefficients
#            self.quad_wts = quad_wts
            pass
        else:
#            print basis.elements
#            print quadrature_points
#            print polynomial_evals
            self.weighted_evals = polynomial_evals * coefficients
#            print self.weighted_evals
            self.quad_wts = quadrature_weights
#            print sum(self.quad_wts)
        self.skewness = getSkewness(self.quad_wts, self.weighted_evals, self.basis, self.variance)
        self.kurtosis = getKurtosis(self.quad_wts, self.weighted_evals, self.basis, self.variance)
        
    
    def plot(self, filename=None):
        """
        Produces a bar graph of the first order Sobol indices

        :param Statistics object: An instance of the Statistics class.
        :param string filename: A file name in case the user wishes to save the bar graph. The default output is an eps file.

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        # A bar graph plot of the first order Sobol indices!
        x = range(len(self.getSobol(1).keys()))
        barplot(x, self.getSobol(1).values(), 'Parameters', 'Sobol indices', self.getSobol(1).keys())
    
    def getSobol(self, order = 1):
        """
        Get Sobol' indices at specified order. 

        :param order int: The order at which Sobol' indices are computed. By default, computes first order Sobol' indices.
        :return: indices, Dictionary where keys specify non-zero dimensions and values represent Sobol' indices.
        :rtype: dict

        **Sample usage:**
        stats = Statistics(coeffcients, basis)
        fosi = stats.getSobol(1)        
        
        """
        return {key: value for key, value in self.sobol.iteritems() if len(key) == order}
        
        
    def getCondSkewness(self, order = 1):
        """
        Get conditional skewness indices at specified order. 

        :param order int: The order at which conditional skewness indices are computed. By default, computes first order conditional skewness.
        :return: indices, Dictionary where keys specify non-zero dimensions and values represent conditional skewness indices.
        :rtype: dict

        **Sample usage:**
        stats = Statistics(coeffcients, basis)
        first_order_skewness = stats.getCondSkewness(1)        
        
        """
        return CondSkewness(order, self.quad_wts, self.weighted_evals, self.basis, self.variance, self.skewness)
    def getCondKurtosis(self, order = 1):
        """
        Get conditional kurtosis indices at specified order. 

        :param order int: The order at which conditional kurtosis indices are computed. By default, computes first order conditional kurtosis.
        :return: indices, Dictionary where keys specify non-zero dimensions and values represent conditional kurtosis indices.
        :rtype: dict

        **Sample usage:**
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
    
    #plots variance, skewness and kurtosis. Users can "update" the input dictionaries first to 
    #plot multiple orders at once
    @staticmethod
    def plot_all_indices(list_of_indices_dicts):       
        assert(len(list_of_indices_dicts) == 3) #v + s + k 
        v = list_of_indices_dicts[0]
        s = list_of_indices_dicts[1]
        k = list_of_indices_dicts[2]
        a = range(len(v))
        b = [x for _,x in sorted(zip(v.keys(),a),key = lambda pair:len(pair[0]))]
        vvals = [x for _,x in sorted(zip(v.keys(),v.values()), key = lambda pair:(len(pair[0]), pair[0][0]))]
        svals = [x for _,x in sorted(zip(v.keys(), s.values()), key = lambda pair:(len(pair[0]), pair[0][0]))]
        kvals = [x for _,x in sorted(zip(v.keys(), k.values()), key = lambda pair:(len(pair[0]), pair[0][0]))]
        triplebarplot(a, vvals, svals, kvals, "Dimensions", "Index Value", sorted(v.keys(), key = len))
    
    #Pie chart of variance, skewness and kurtosis indices
    #Var names in list form
    @staticmethod
    def pie_chart( list_of_indices_dicts, highest_order = 1, var_names = None, title = "Sobol' indices"):
        v = list_of_indices_dicts[0]
        if len(list_of_indices_dicts) > 1:
            s = list_of_indices_dicts[1]
        if len(list_of_indices_dicts) > 2:
            k = list_of_indices_dicts[2]
        labels_and_values = {}
        if not(var_names is None):
            v = dict((tuple([var_names[i] for i in key]), val) for key, val in v.iteritems())
        for i in v.keys():
            if v[i] > 1e-6:
                if len(i) <= highest_order:
                    labels_and_values[i] = v[i]
                else:
                    key = "order " + str(len(i))
                    try:
                        labels_and_values[key] += v[i]
                    except KeyError:
                        labels_and_values[key] = v[i]
                    
#        for i in s.keys():
#            if len(i) == 1:
#                labels_and_values[i[0]] = s.values[i]
#            else:
#                key = "order " + str(len(i))
#                labels_and_values[key] += s.values[i]
#        for i in k.keys():
#            if len(i) == 1:
#                labels_and_values[i[0]] = k.values[i]
#            else:
#                key = "order " + str(len(i))
#                labels_and_values[key] += k.values[i]
        
        labels = labels_and_values.keys()
        values = labels_and_values.values()
        vl = sorted(zip(values, labels), reverse = True)
        values, labels = zip(*vl)
        
        piechart(labels, values, title)
        
    @staticmethod
    def scatter_plot(list_of_indices_dicts, highest_order = 2, var_names = None, title = "Sobol' indices"):
        # Assume all dicts have the same keys!
        list_of_dicts = list_of_indices_dicts[:]
        if not(var_names is None):
            for j in range(len(list_of_indices_dicts)):
                list_of_dicts[j] = dict((tuple([var_names[i] for i in key]), val) for key, val in list_of_indices_dicts[j].iteritems())
        
        valid_keys = [i for i in list_of_dicts[0].keys() if len(i) <= highest_order]
        
        labels = sorted(valid_keys, key = len)

        col = np.reshape(np.arange(len(valid_keys), dtype = 'float'), (len(valid_keys),1))
        list_of_cols = []
        for i in range(len(list_of_dicts)):
            list_of_cols.append(col.copy())
        x = np.hstack(list_of_cols)
        y = x.copy()
        for i in range(len(list_of_dicts)):
            for j in range(y.shape[0]):
                y[j,i] = list_of_dicts[i][labels[j]]
        
        
        scatterplot2(x,y,labels)
                
            
        
            
            
        
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
def getAllSobol(coefficients, basis):
    m, n = coefficients.shape
    variance = getVariance(coefficients)
    if m > n:
        coefficients = coefficients.T
    
    if not(isinstance(basis, np.ndarray)):
        basis = basis.elements
    m, dimensions = basis.shape
    if dimensions == 1:
        return {0:1.0}
    else:
        basis_entries = m

        #Build dict to contain the Sobol' indices
        #Keys contain the non-zero indices
        #Values contain the Sobol' indices
                
        combo_index = {}
        for order in range(1,dimensions+1): #loop over order            
            for i in combinations(range(dimensions),order):
                #initialize each index to be 0                
                combo_index[i] = 0
                
            for i in range(0,basis_entries): #loop over rows
                row = basis[i,:]
                non_zero_entries = np.nonzero(row)[0]
                non_zero_entries.sort()    #just in case
                if len(non_zero_entries) == order: #neglect entries that should actually be zero (what constitutes as zero?)
                    combo_index[tuple(non_zero_entries)] = combo_index[tuple(non_zero_entries)] + coefficients[0][i]**2 / variance
        
        check_sum = sum(combo_index.values())
        if (abs(check_sum - 1.0) >= 1e-2):
            print "Possible discrepancy in calculation, sum of indices = " + str(check_sum) 
        
        return combo_index

# Return global skewness        
def getSkewness(quad_wts, weighted_evals, basis, variance):    
    total_evals = np.sum(weighted_evals[1:],0)
#    print weighted_evals[0]
    third_total_evals = total_evals**3
    
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
    norm_ind = map(tuple,(norm_ind > 0).astype(int))
    
    combo_index = {}
    for tot_order in range(1,dimensions+1): #loop over order            
        for i in product([0,1], repeat = dimensions):
            #initialize each index of the specified order to be 0                
            if sum(i) != order:
                continue
            combo_index[i] = 0.0   
    
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
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.iteritems()}
    return combo_index

# Return conditional kurtosis of specified order, in dictionary format similar to Sobol' indices
def CondKurtosis(order, quad_wts, weighted_evals, basis, variance, kurtosis):
    #Get all polynomials evaluated at the quad. pts and corresponding wts
    dimensions = basis.elements.shape[1]
    norm_ind = basis.elements.copy()
    norm_ind = map(tuple,(norm_ind > 0).astype(int))
    
    combo_index = {}
    for tot_order in range(1,dimensions+1): #loop over order            
        for i in product([0,1], repeat = dimensions):
            #initialize each index to be 0                
            if sum(i) != order:
                continue
            combo_index[i] = 0.0   
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
    combo_index = {tuple(np.nonzero(key)[0]): value for key, value in combo_index.iteritems()}
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

