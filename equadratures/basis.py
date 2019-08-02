"Rountines for defining the index set associated with multivariate polynomials."
import numpy as np
import math as mt

class Basis(object):
    """
    Basis class constructor.

    :param string basis_type: The type of index set to be used. Options include:
        `Univariate', `Total order`, `Tensor grid`, `Sparse grid`, `Hyperbolic basis` and `Euclidean degree`. All basis are isotropic. If you require anisotropic basis do email us.
    :param ndarray orders: List of integers corresponding to the highest polynomial order in each direction.
    :param string growth_rule: The type of growth rule associated with sparse grids. Options include: `linear' and `exponential'. This input is only required when using a sparse grid.
    :param double q: The `q' parameter is used to control the number of basis terms used in a hyperbolic basis. It varies between 0.0 to 1.0. A value of 1.0 yields a total order basis.

    Attributes:
        * **self.dimension**: (integer) Number of dimensions of the index set.
        * **self.elements**: (numpy array) The multi-indices of the index set.
        * **self.cardinality**:(integer) The cardinality of the index set.

    **Notes:**

    For details on the Euclidean degree see: `Trefethen 2016 <https://arxiv.org/pdf/1608.02216v1.pdf>`_.
    Note that all index sets are sorted in the constructor automatically, by their total orders. We will be adding non-isotropic index sets in a future release. Stay tuned!

    """
    def __init__(self, basis_type, orders=None, level=None, growth_rule=None, q=None):

        # Required
        self.basis_type = basis_type # string

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

        # Orders
        if orders is None:
            self.orders = []
        else:
            self.set_orders(orders)
    def set_orders(self, orders):
        """
        Sets the highest order in each direction of the basis.

        :param Basis object: An instance of the Basis class.
        :param list orders: The highest polynomial order along each dimension.

        """
        self.orders = []
        for i in range(0, len(orders)):
            self.orders.append(orders[i])
        self.dimension = len(self.orders)
        name = self.basis_type
        if name.lower() == "total-order":
            basis = total_order_basis(self.orders)
        elif name.lower() ==  "univariate":
            basis = np.reshape( np.linspace(0, self.orders[0], self.orders[0]+1) , (self.orders[0]+1, 1) )
        elif name.lower() == "sparse-grid":
            sparse_index, a, SG_set = sparse_grid_basis(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            basis = SG_set
        elif (name.lower() == "tensor-grid") or (name.lower() == "tensor") :
            basis = tensor_grid_basis(self.orders)
        elif name.lower() == "hyperbolic-basis":
            basis = hyperbolic_basis(self.orders, self.q)
        elif name.lower() == "euclidean-degree":
            basis = euclidean_degree_basis(self.orders)
        else:
            raise(ValueError, 'Basis __init__: invalid value for basis_type!')
            basis = [0]
        self.elements = basis
        self.cardinality = len(basis)
    def prune(self, number_of_elements_to_delete):
        """
        Prunes down the number of elements in an index set.

        :param Basis object: An instance of the Basis class.
        :param integer number_of_elements_to_delete: The number of multi-indices the user would like to delete.

        """
        index_entries = self.elements
        total_elements = self.cardinality
        new_elements = total_elements - number_of_elements_to_delete
        if new_elements < 0 :
            raise(ValueError, 'In Basis() --> prune(): Number of elements to be deleted must be greater than the total number of elements')
        else:
            self.elements =  index_entries[0:new_elements, :]
    def sort(self):
        """
        Routine that sorts a multi-index in ascending order based on the total orders. The constructor by default calls this function.

        :param Basis object: An instance of the Basis class.
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
    def get_basis(self):
        """
        Gets the index set elements for the Basis object.

        :param Basis object: An instance of the Basis class.
        """
        name = self.basis_type
        if name == "total-order":
            basis = total_order_basis(self.orders)
        elif name == "tensor-grid":
            basis = tensor_grid_basis(self.orders)
        elif name == "hyperbolic-basis":
            basis = hyperbolic_basis(self.orders, self.q)
        elif name == "euclidean-degree":
            basis = euclidean_degree_basis(self.orders)
        elif name == "sparse-grid":
            sparse_index, sparse_weight_factors, sparse_grid_set = sparse_grid_basis(self.level, self.growth_rule, self.dimension) # Note sparse grid rule depends on points!
            return sparse_index, sparse_weight_factors, sparse_grid_set
        else:
            raise(ValueError, 'invalid value for basis_type!')
            basis = [0]
        return basis



#---------------------------------------------------------------------------------------------------
# PRIVATE FUNCTIONS
#---------------------------------------------------------------------------------------------------
def euclidean_degree_basis(orders):
    dimensions = len(orders)
    n_bar = tensor_grid_basis(orders)
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

def hyperbolic_basis(orders, q):

    # Initialize a few parameters for the setup
    dimensions = len(orders)
    n_bar = total_order_basis(orders)
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
def getTotalOrderBasisRecursion(highest_order, dimensions):
   if dimensions == 1:
       I = np.zeros((1,1))
       I[0,0] = highest_order
   else:
       for j in range(0, highest_order + 1):
           U = getTotalOrderBasisRecursion(highest_order - j, dimensions - 1)
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

def total_order_basis(orders):
    dimensions = len(orders)
    highest_order = np.max(orders)
    total_order = np.zeros((1, dimensions))
    for i in range(1, highest_order+1):
        R = getTotalOrderBasisRecursion(i, dimensions)
        total_order = np.vstack((total_order, R))
    return total_order
def sparse_grid_basis(level, growth_rule, dimensions):

    # Initialize a few parameters for the setup
    level_new = level - 1
    lhs = int(level_new) + 1
    rhs = int(level_new) + dimensions

    # Set up a global tensor grid
    tensor_elements = np.ones((dimensions))
    for i in range(0, dimensions):
        tensor_elements[i] = int(rhs)

    n_bar = tensor_grid_basis(tensor_elements) #+ 1

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
        k = int(level_new + dimensions - summation2[i])
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
        SG_indices[i] = tensor_grid_basis(sparse_index[i,:] )
        counter = counter + len(SG_indices[i])

    SG_set = np.zeros((counter, dimensions))
    counter = 0
    for i in range(0, len(sparse_index)):
        for j in range(0, len(SG_indices[i]) ):
            SG_set[counter,:] = SG_indices[i][j]
            counter = counter + 1
    return sparse_index, a, SG_set

def tensor_grid_basis(orders):

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
    basis = I[:,1::]
    return basis

def column(matrix, i):
    return [row[i] for row in matrix]
