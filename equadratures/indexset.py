#!/usr/bin/env python
"""Index sets for multivariate polynomials"""
import numpy as np
import math as mt
from .plotting import scatterplot, scatterplot3D

class IndexSet(object):
    """
    :param string index_set_type: The type of index set to be used. Options include:
        `Total order`, `Tensor grid`, `Sparse grid`, `Hyperbolic basis` and `Euclidean degree`. All basis are isotropic. If you require anisotropic basis do email us.
    :param ndarray orders: List of integers corresponding to the highest polynomial order in each direction.
    :param string growth_rule: The type of growth rule associated with sparse grids. Options include: `linear' and `exponential'. This input is only required when using a sparse grid.
    :param integer dimensions: The number of dimensions of the problem. This input is only required when using a sparse grid.
    :param double q: The `q' parameter is used to control the number of basis terms used in a hyperbolic basis. It varies between 0.0 to 1.0. A value of 1.0 yields a total order basis. 
    
    Attributes:
        * **self.dimension**: (integer) Number of dimensions of the index set.
        * **self.elements**: (numpy array) The multi-indices of the index set.
        * **self.cardinality**:(integer) The cardinality of the index set.

    **Notes:** 
    
    For details on the Euclidean degree see: `Trefethen 2016 <https://arxiv.org/pdf/1608.02216v1.pdf>`_. 
    Note that all index sets are sorted in the constructor automatically, by their total orders. We will be adding non-isotropic index sets in a future release. Stay tuned!

    **Sample usage** 
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

        # For hyperbolic basis index set, there is a "q" parameter:
        if q is None:
            self.q = []
        else:
            self.q = q

        # Check for problem dimensionality (only for sparse grids)
        if dimension is None:
            self.dimension = len(orders)
        else:
            self.dimension = dimension

        name = self.index_set_type
        if name == "Total order":
            index_set = total_order_index_set(self.orders)
        elif name == "Sparse grid":
            sparse_index, a, SG_set = sparse_grid_index_set(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            index_set = SG_set
        elif name == "Tensor grid":
            index_set = tensor_grid_index_set(self.orders)
        elif name == "Hyperbolic basis":
            index_set = hyperbolic_index_set(self.orders, self.q)
        elif name == "Euclidean degree":
            index_set = euclidean_degree_index_set(self.orders)
        else:
            raise(ValueError, 'indexset __init__: invalid value for index_set_type!')
            index_set = [0]
        
        self.elements = index_set
        self.cardinality = len(index_set)

        # Don't sort a sparse grid index set, because the order is tied to the coefficients!
        if name =="Hyperbolic basis":
            self.sort()
        elif name == "Euclidean degree":
            self.sort()
        elif name == "Total order":
            self.sort()

    def prune(self, number_of_elements_to_delete):
        """
        Prunes down the number of elements in an index set 

        :param IndexSet object: An instance of the IndexSet class.
        :param integer number_of_elements_to_delete: The number of multi-indices the user would like to delete

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        index_entries = self.elements
        total_elements = self.cardinality
        new_elements = total_elements - number_of_elements_to_delete
        if new_elements < 0 :
            raise(ValueError, 'In IndexSet() --> prune(): Number of elements to be deleted must be greater than the total number of elements')
        else:
            self.elements =  index_entries[0:new_elements, :]
    
    def sort(self):
        """
        Routine that sorts a multi-index in ascending order based on the total orders. The constructor by default calls this function.

        :param IndexSet object: An instance of the IndexSet class.

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        number_of_elements = len(self.elements)
        combined_indices_for_sorting = np.ones((number_of_elements, 1))
        sorted_elements = np.ones((number_of_elements, self.dimension))
        elements = self.elements
        for i in range(0, number_of_elements):
            a = np.sort(elements[i,:])
            u = 0
            for j in range(0, self.dimension):
                u = 10**(j) * a[j] + u
            combined_indices_for_sorting[i] = u
        sorted_indices = np.argsort(combined_indices_for_sorting, axis=0)
       
       # Create a new index set with the sorted entries
        for i in range(0, number_of_elements):
            for j in range(0, self.dimension):
                row_index = sorted_indices[i]
                sorted_elements[i,j] = elements[row_index, j]
        self.elements = sorted_elements

    def getIndexSet(self):
        """
        Prunes down the number of elements in an index set 

        :param IndexSet object: An instance of the IndexSet class.
        :param integer number_of_elements_to_delete: The number of multi-indices the user would like to delete

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
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
        elif name == "Euclidean degree":
            index_set = euclidean_degree_index_set(self.orders)
        else:
            raise(ValueError, 'indexset __init__: invalid value for index_set_type!')
            index_set = [0]
        
        return index_set
    
    def plot(self, filename=None):
        """
        Plots the index set

        :param IndexSet object: An instance of the IndexSet class.
        :param string filename: A file name in case the user wishes to save the bar graph. The default output is an eps file.

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        elements = self.elements
        elements = np.mat(elements)

        if self.dimension == 2:
            scatterplot(elements[:,0], elements[:,1], r'$j_1$', r'$j_2$', filename)
        
        elif self.dimension == 3:
            scatterplot3D( elements[:,[0,1]]  , elements[:,2] , r'$j_1$', r'$j_2$', r'$j_3$', filename)

        else:
            raise(ValueError, 'IndexSet()->plot(): Can only generate plots when the dimension is 2 or 3!')


#---------------------------------------------------------------------------------------------------
# PRIVATE FUNCTIONS
#---------------------------------------------------------------------------------------------------
def euclidean_degree_index_set(orders):
    dimensions = len(orders)
    n_bar = tensor_grid_index_set(orders)
    n_new = []
    l2norms = np.ones((1, len(n_bar)))
    for i in range(0, len(n_bar)):
        array_entry = n_bar[i]**2
        l2norms[0,i] = np.sum(array_entry)  # dimension correction!

    maxval = np.max(np.array(orders)**2)
    for i in range(0, len(l2norms[0,:])):
        if( l2norms[0,i] <= maxval):
            n_new.append(n_bar[i,:])

    # Now re-cast n_new as a regular array and not a list!
    euclidean_set = np.ones((len(n_new), dimensions))
    for i in range(0, len(n_new)):
        for j in range(0, dimensions):
            r = n_new[i]
            euclidean_set[i,j] = r[j]
    
    return euclidean_set


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
def getTotalOrderIndexSetRecursion(highest_order, dimensions):    
   if dimensions == 1:
       I = np.zeros((1,1))
       I[0,0] = highest_order
   else:
       for j in range(0, highest_order + 1):
           U = getTotalOrderIndexSetRecursion(highest_order - j, dimensions - 1)
           rows, cols = U.shape
           T = np.zeros((rows, cols + 1) ) # allocate space!
           T[:,0] = j * np.ones((1, rows))
           T[:, 1: cols+1] = U
           if j == 0:
               I = T
           elif j >= 0:
               rows_I, cols_I = I.shape
               rows_T, cols_T = T.shape
               Itemp = np.zeros((rows_I + rows_T, cols_I))
               Itemp[0:rows_I,:] = I
               Itemp[rows_I : rows_I + rows_T, :] = T
               I = Itemp
           del T
   return I


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

    #print 'n_new'
    #print n_new
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
                sparse_index[i,j] = int(2**(r[j] - 1)  )
            elif(growth_rule is 'linear'):
                sparse_index[i,j] = int(r[j])
            else:
                raise KeyboardInterrupt
    
    #print sparse_index
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
        vector_of_ones_a = np.ones((int(orders[u]+1), 1))
        vector_of_ones_b = np.ones((len(I), 1))
        counting = np.arange(0,orders[u]+1)
        counting = np.reshape(counting, (len(counting), 1) )
        left_side = np.array(np.kron(I, vector_of_ones_a))
        right_side = np.array( np.kron(vector_of_ones_b, counting) )  # make a row-wise vector
        I =  np.concatenate((left_side, right_side), axis = 1)

    # Ignore the first column of pp
    index_set = I[:,1::]

    return index_set

def column(matrix, i):
    return [row[i] for row in matrix]
