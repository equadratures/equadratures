#!/usr/bin/env python
"""Index sets for multivariate polynomials"""
import numpy as np
import math as mt
from utils import error_function

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
        >> IndexSet('Sparse grid', level=3, growth_rule=5, dimension=3)
        >> IndexSet('Total order' [3, 3, 3])
        >> IndexSet('Hyperbolic basis', [3,3], q=0.75)

    """
    def __init__(self, index_set_type, orders=None, level=None, growth_rule=None, dimension=None, q=None):
        
        # Required
        self.index_set_type = index_set_type # string
        
        # Orders
        if orders is None:
            self.orders = []
        else:
            self.orders = orders
        
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

        # For hyperbolic basis index set, there is a "q" parameter:
        if q is None:
            self.q = []
        else:
            self.q = q

        name = self.index_set_type
        if name == "Total order":
            index_set = total_order_index_set(self.orders)
        elif name == "Sparse grid":
            sparse_index, a, SG_set = sparse_grid_index_set(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            return sparse_index, a, SG_set
        elif name == "Tensor grid":
            index_set = tensor_grid_index_set(self.orders)
        elif name == "Hyperbolic basis":
            index_set = hyperbolic_index_set(self.orders, self.q)
        else:
            error_function('indexset __init__: invalid value for index_set_type!')
            index_set = [0]
        
        self.elements = index_set

    def prune(self, number_of_elements_to_delete):
        """
        Prunes down the number of elements in an index set!
        > shouldn't this be somehow done by the total order or sth??
        """
        index_entries = self.elements
        total_elements = self.getCardinality()
        new_elements = total_elements - number_of_elements_to_delete
        if new_elements < 0 :
            raise(ValueError, 'In IndexSet() --> prune(): Number of elements to be deleted must be greater than the total number of elements')
        else:
            self.elements =  index_entries[0:new_elements, :]
    
    def getCardinality(self):
        """
        Returns the cardinality (total number of elements) of the index set

        :param IndexSet self: An instance of the IndexSet class

        :return: cardinality: Total number of elements in the index set
        :rtype: integer
        """
        name = self.index_set_type
        if name == "Total order":
            index_set = total_order_index_set(self.orders)
        elif name == "Sparse grid":
            index_set, a, SG_set = sparse_grid_index_set(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            return sparse_index, a, SG_set
        elif name == "Tensor grid":
            index_set = tensor_grid_index_set(self.orders)
        elif name == "Hyperbolic basis":
            index_set = hyperbolic_index_set(self.orders, self.q)
        else:
            error_function('indexset __init__: invalid value for index_set_type!')

        # Now m or n is equivalent to the
        return len(index_set)

    def getIndexSet(self):
        """
        Returns all the elements of an index set

        :param IndexSet self: An instance of the IndexSet class

        :return: iset: An n-by-d array of index set elements. Here n represents the cardinality of the index set
            while d represents the number of dimensions.
        :rtype: ndarray
        """
        name = self.index_set_type
        if name == "Total order":
            index_set = total_order_index_set(self.orders)
        elif name == "Sparse grid":
            sparse_index, a, SG_set = sparse_grid_index_set(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            return sparse_index, a, SG_set
        elif name == "Tensor grid":
            index_set = tensor_grid_index_set(self.orders)
        elif name == "Hyperbolic basis":
            index_set = hyperbolic_index_set(self.orders, self.q)
        else:
            error_function('indexset __init__: invalid value for index_set_type!')
            index_set = [0]
        
        return index_set

# If this works correctly, this function should sort an index based on the total order of elements.
def sortIndices(index_set_elements):
    elements = len(index_set_elements)
    dims = len(index_set_elements[0,:])
    highest_orders_per_row = np.zeros((elements, 1))
    for i in range(0, elements):
        highest_orders_per_row[i,0] = np.max(elements[i,:])
    
    notused, sorted_indices = np.sort(highest_orders_per_row)
    index_set_elements = index_set_elements[sorted_indices, :]
    highest_order = np.max(highest_orders_per_row)

    # If there are repeats, then sort them by the sum of all indices!
    allorders = np.arange(0,highest_order)
        
        # Find all the repeat values for the highest order
        for i in range(0, highest_order):
            ii = np.where(elements == allorders[i])
            if ii is not None:
                for j in range(0, len(ii)):

    
def getIndexLocation(small_index, large_index):
    index_values = []
    i = 0
    while i < len(small_index):
        for j in range(0, len(large_index)):
            if np.array_equal(small_index[i,:] , large_index[j,:] ):
                index_values.append(j)
        i = i + 1

    return index_values

def hyperbolic_index_set(orders, q):

    # Initialize a few parameters for the setup
    dimensions = len(orders)
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
