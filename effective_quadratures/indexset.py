#!/usr/bin/env python
"""Index sets for multivariate polynomials"""
import numpy as np
import math as mt

class IndexSet(object):
    
    """
    This class defines an index set

   :param string index_set_type: The type of index set to be used. Options include:
        `Total order`, `Tensor grid`, `Sparse grid` and `Hyperbolic basis`. 
   :param ndarray of integers orders: The highest polynomial order in each direction
   :param integer level: For sparse grids the level of the sparse grid rule
   :param string growth_rule: The type of growth rule associated with the sparse grid. Options include:
        `linear` and `exponential`.


    **Sample declarations** 
    ::
        
        >> IndexSet('Tensor grid', [3,3,3])
        >> IndexSet('Sparse grid', )


    """
    def __init__(self, index_set_type, orders, level=None, growth_rule=None, dimension=None):
        
        self.index_set_type = index_set_type # string
        self.orders =  orders # we store order as an array!

        # Check for the levels (only for sparse grids)
        if level is None:
            self.level = []
        else:
            self.level = level

        # Check for the growth rule (only for sparse grids)
        if growth_rule is None:
            self.growth_rule = []
        else:
            self.growth_rule = growth_rule

        # Check for problem dimensionality (only for sparse grids)
        if dimension is None:
            self.dimension = []
        else:
            self.dimension = dimension

    def getCardinality(self):
        """
        Returns the cardinality (total number of elements) of the index set

        :param Parameter self: An instance of the Parameter class
        :param int order: Number of eigenvectors required. This function makes the call getJacobiMatrix(order) and then computes
            the corresponding eigenvectors.

        :return: V, order-by-order matrix that contains the eigenvectors of the Jacobi matrix
        :rtype: ndarray

        **Sample declaration**
        :: 
            # Code to Jacobi eigenvectors
            >> var4 = Parameter(points=5, param_type='Gaussian', shape_parameter_A=0, shape_parameter_B=2)
            >> V = var4.getJacobiEigenvectors()
        """
        if self.index_set_type == "sparse grid":
            index_set, a, SG_set = getindexsetvalues(self)
        else:
            index_set = getindexsetvalues(self)
            return len(index_set)

        # Now m or n is equivalent to the
        return len(index_set)

    def getIndexSetType(self):
        return self.index_set_type

    def getOrders(self):
        return self.orders

    def getMaxOrders(self):
        return self.orders

    def getIndexSet(self):
        return getindexsetvalues(self)

    def determineIndexLocation(self, large_index):
        small_index = getindexsetvalues(self)
        return getIndexLocation(small_index, large_index)

# From a large index determine the specific locations of entries of the smaller index set.
# This functionality will be used for error computation. We assume that elements in both
# small_index and large_index are unique.
def getIndexLocation(small_index, large_index):
    index_values = []
    i = 0
    while i < len(small_index):
        for j in range(0, len(large_index)):
            if np.array_equal(small_index[i,:] , large_index[j,:] ):
                index_values.append(j)
        i = i + 1

    return index_values


def getindexsetvalues(self):

    name = self.index_set_type
    if name == "Total order":
        index_set = total_order_index_set(self.orders)
    elif name == "Sparse grid":
        sparse_index, a, SG_set = sparse_grid_index_set(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
        return sparse_index, a, SG_set
    elif name == "Tensor grid":
        index_set = tensor_grid_index_set(self.orders )
    elif name == "Hyperbolic basis":
        index_set = hyperbolic_index_set(self.orders, self.level)
    else:
        index_set = [0]
    return index_set


def hyperbolic_index_set(orders, q):

    # Initialize a few parameters for the setup
    dimensions = len(orders)
    #for i in range(0, dimensions):
    #    orders[i] = orders[i] - 1


    n_bar = tensor_grid_index_set(orders)
    n_new = []
    summation = np.ones((1, len(n_bar)))
    for i in range(0, len(n_bar)):
        array_entry = n_bar[i] ** q
        summation[0,i] = ( np.sum(array_entry)  ) ** (1.0/(1.0 * q)) # dimension correction!

    # Loop!
    for i in range(0, len(summation[0,:])):
        if( summation[0,i] <= np.max(orders) ):
            n_new.append(n_bar[i,:])

    # Now re-cast n_new as a regular array and not a list!
    hyperbolic_set = np.ones((len(n_new), dimensions))
    for i in range(0, len(n_new)):
        for j in range(0, dimensions):
            r = n_new[i]
            hyperbolic_set[i,j] = r[j]

    return hyperbolic_set

# Double checked April 7th, 2016 --> Works!
def total_order_index_set(orders):

    # For a total order index set, the sum of all the elements of a particular
    # order cannot exceed that of the polynomial.

    # Initialize a few parameters for the setup
    dimensions = len(orders)
    n_bar = tensor_grid_index_set(orders)
    n_new = [] # list; dynamic array

    # Now cycle through each entry, and check the sum
    summation = np.sum(n_bar, axis=1)
    for i in range(0, len(summation)):
        if(summation[i]  <= np.max(n_bar) ):
            value = n_bar[i,:]
            n_new.append(value)

    # But I want to re-cast this list as an array
    total_index = np.ones((len(n_new), dimensions))
    for i in range(0, len(n_new)):
        for j in range(0, dimensions):
            r = n_new[i]
            total_index[i,j] = r[j]

    return total_index



def sparse_grid_index_set(level, growth_rule, dimensions):

    # Initialize a few parameters for the setup
    lhs = int(level) + 1
    rhs = int(level) + dimensions

    # Set up a global tensor grid
    tensor_elements = np.ones((dimensions))
    for i in range(0, dimensions):
        tensor_elements[i] = int(rhs)

    n_bar = tensor_grid_index_set(tensor_elements) + 1

    # Check constraints
    n_new = [] # list; a dynamic array
    summation = np.sum(n_bar, axis=1)
    for i in range(0, len(summation)):
        if(summation[i] <= rhs  and summation[i] >= lhs):
            value = n_bar[i,:]
            n_new.append(value)

    # Sparse grid coefficients
    summation2 = np.sum(n_new, axis=1)
    a = [] # sparse grid coefficients
    for i in range(0, len(summation2)):
        k = int(level + dimensions - summation2[i])
        n = int(dimensions -1)
        value = (-1)**k  * (mt.factorial(n) / (1.0 * mt.factorial(n - k) * mt.factorial(k)) )
        a.append(value)

    # Now sort out the growth rules
    sparse_index = np.ones((len(n_new), dimensions))
    for i in range(0, len(n_new)):
        for j in range(0, dimensions):
            r = n_new[i]
            if(r[j] - 1 == 0):
                sparse_index[i,j] = int(1)
            elif(growth_rule is 'exponential' and  r[j] - 1 != 0 ):
                sparse_index[i,j] = int(2**(r[j] - 1) + 1 )
            elif(growth_rule is 'linear'):
                sparse_index[i,j] = int(r[j])
            else:
                raise KeyboardInterrupt

    # Ok, but sparse_index just has the tensor order sets to be used. Now we need
    # to get all the index sets!
    SG_indices = {}

    counter = 0
    for i in range(0, len(sparse_index)):
        SG_indices[i] = tensor_grid_index_set(sparse_index[i,:] )
        counter = counter + len(SG_indices[i])

    SG_set = np.zeros((counter, dimensions))
    counter = 0
    for i in range(0, len(sparse_index)):
        for j in range(0, len(SG_indices[i]) ):
            SG_set[counter,:] = SG_indices[i][j]
            counter = counter + 1
    return sparse_index, a, SG_set

def tensor_grid_index_set(orders):

    dimensions = len(orders) # number of dimensions
    I = [1.0] # initialize!

    # For loop across each dimension
    for u in range(0,dimensions):

        # Tensor product of the points
        vector_of_ones_a = np.ones((orders[u]+1, 1))
        vector_of_ones_b = np.ones((len(I), 1))
        counting = np.arange(0,orders[u]+1)
        counting = np.reshape(counting, (len(counting), 1) )
        left_side = np.array(np.kron(I, vector_of_ones_a))
        right_side = np.array( np.kron(vector_of_ones_b, counting) )  # make a row-wise vector
        I =  np.concatenate((left_side, right_side), axis = 1)

    # Ignore the first column of pp
    index_set = I[:,1::]

    return index_set

def error_function(string_value):
    print string_value
    sys.exit()
