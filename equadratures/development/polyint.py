"""Operations involving multivariate polynomials (without gradients) via numerical quadrature"""
#from .parameter import Parameter
import numpy as np
from math import factorial
from itertools import combinations
from .utils import evalfunction, find_repeated_elements, meshgrid, removeDuplicates
from .plotting import bestfit, bestfit3D, histogram
from .qr import solveLSQ
#from .stats import Statistics

class Polyint(Poly):
    """
    This class defines a Polyint (polynomial via integration) object

    :param array of Parameters uq_parameters: A list of Parameters
    :param IndexSet index_set: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no index_set parameter input is given.
    
    **Sample declarations** 
    ::
        >> s = Parameter(lower=-2, upper=2, param_type='Uniform', points=4)
        >> T = IndexSet('Total order', [3,3])
        >> polyObject = Polynomial([s,s],T) # basis is defined by T

        >> s = Parameter(lower=-2, upper=2, param_type='Uniform')
        >> polyObject = Polynomial([s,s]) # Tensor basis is used
    """

    # Constructor
    def __init__(self, parameters, basis, sampling=Nonem fun=None):
        super(Polyint, self).__init__(parameters, basis)
        if sampling is None:
            sampling = 'tensor grid quadrature'
        self.setSamplingMethod()
    
    @staticmethod
    def setSamplingMethod(self):
        if not(self.sampling.lower() in ["tensor grid quadrature", "sparse grid quadrature", "effectively subsampled quadrature",
            "Christoffel subsampled", "induced distribution samples", "randomized quadrature"]) :
            raise(ValueError, 'Polyint:generatePointsForEvaluation:: Sampling method not defined! Choose from existing ones.') 
        if sampling.lower() == 'tensor grid quadrature':
            points, weights = self.tensor_grid_quadrature()
        elif sampling.lower() == 'sparse grid quadrature':
            points, weights = self.sparse_grid_quadrature()
        elif sampling.lower() == 'effectively subsampled quadrature':
            points, weights = self.effective_quadrature()
        elif sampling.lower() == 'randomized quadrature'
            points, weights = self.randomized_quadrature()
        elif sampling.lower() == 'Christoffel subsampled':
            points, weights = self.christoffel_quadrature()
        elif sampling.lower() == 'induced distribution samples'
        self.points = points
        self.weights = weights


    def tensor_grid_quadrature(self):

    def computeCoefficients(self):
        if sampling.lower() == 'tensor grid':
            self.coefficients, self.basis_elements, 

    def getPolynomialCoefficients(self, function):  
        """
        Returns multivariate orthonormal polynomial coefficients. Depending on the choice of the index set, this function will either return a tensor grid
        of pseudospectral coefficients or a sparse grid using the SPAM technique by Constantine et al (2012). 
    
        :param Polynomial self: An instance of the Polynomial class
        :param: callable function: The function that needs to be approximated (or interpolated)
        :return: coefficients: The pseudospectral coefficients
        :rtype: ndarray
        :return: indexset: The indices used for the pseudospectral computation
        :rtype: ndarray
        :return: evaled_pts: The points at which the function was evaluated
        :rtype: ndarray

        """
        # Method to compute the coefficients
        method = self.index_sets.index_set_type
        # Get the right polynomial coefficients
        if method == "Tensor grid":
            
        if method == "Sparse grid":
            coefficients, indexset, evaled_pts = getSparsePseudospectralCoefficients(self, function)
        else:
            coefficients, indexset, evaled_pts = getPseudospectralCoefficients(self, function)
        return coefficients,  indexset, evaled_pts

#--------------------------------------------------------------------------------------------------------------
#
#  PRIVATE FUNCTIONS!
#
#--------------------------------------------------------------------------------------------------------------
def tensorgrid(stackOfParameters, function=None):
    """
    Computes a tensor grid of quadrature points based on the distributions for each Parameter in stackOfParameters 

    :param Parameter array stackOfParameters: A list of Parameter objects
    :param callable function: The function whose integral needs to be computed. Can also be input as an array of function values at the
        quadrature points. If the function is given as a callable, then this routine outputs the integral of the function and an array of
        the points at which the function was evaluated at to estimate the integral. These are the quadrature points. In case the function is
        not given as a callable (or an array, for that matter), then this function outputs the quadrature points and weights. 
      
    :return: tensor_int: The tensor grid approximation of the integral
    :rtype: double
    :return: points:  The quadrature points
    :rtype: numpy ndarray
    :return: weights: The quadrature weights
    :rtype: numpy ndarray

    **Notes**
    For further details on this routine, see Polynomial.getPointsAndWeights()

    """
    # Determine the index set to be used!
    dimensions = len(stackOfParameters)
    orders = []
    flags = []
    uniform = 1
    not_uniform = 0
    for i in range(0, dimensions):
        orders.append(stackOfParameters[i].order)
        if stackOfParameters[i].param_type is 'Uniform':
            flags.append(uniform)
        else:
            flags.append(not_uniform)

    tensor = IndexSet('Tensor grid', orders)
    polyObject = Polynomial(stackOfParameters, tensor)

    # Now compute the points and weights
    points, weights = polyObject.getPointsAndWeights()
    
    # For normalizing!
    for i in range(0, dimensions):
        if flags[i] == 0:
            weights  = weights
        elif flags[i] == 1:
            weights = weights * (stackOfParameters[i].upper - stackOfParameters[i].lower )
            weights = weights/(2.0)

    # Now if the function is a callable, then we can compute the integral:
    if function is not None and callable(function):
        tensor_int = np.mat(weights) * evalfunction(points, function)
        return tensor_int, points
    else:
        return points, weights

def sparsegrid(stackOfParameters, level, growth_rule, function=None):
    """
    Computes a sparse grid of quadrature points based on the distributions for each Parameter in stackOfParameters 

    :param Parameter array stackOfParameters: A list of Parameter objects
    :param integer level: Level parameter of the sparse grid integration rule
    :param string growth_rule: Growth rule for the sparse grid. Choose from 'linear' or 'exponential'.
    :param callable function: The function whose integral needs to be computed. Can also be input as an array of function values at the
        quadrature points. If the function is given as a callable, then this routine outputs the integral of the function and an array of
        the points at which the function was evaluated at to estimate the integral. These are the quadrature points. In case the function is
        not given as a callable (or an array, for that matter), then this function outputs the quadrature points and weights. 
      
    :return: sparse_int: The sparse grid approximation of the integral
    :rtype: double
    :return: points:  The quadrature points
    :rtype: numpy ndarray
    :return: weights: The quadrature weights
    :rtype: numpy ndarray

    """
    # Determine the index set to be used!
    dimensions = len(stackOfParameters)
    orders = []
    flags = []
    uniform = 1
    not_uniform = 0
    for i in range(0, dimensions):
        orders.append(stackOfParameters[i].order)
        if stackOfParameters[i].param_type is 'Uniform':
            flags.append(uniform)
        else:
            flags.append(not_uniform)
        
    # Call the sparse grid index set
    sparse = IndexSet('Sparse grid', level=level, growth_rule=growth_rule, dimension=dimensions)
    sparse_index, sparse_coeffs, sparse_all_elements =  sparse.getIndexSet()

    # Get this into an array
    rows = len(sparse_index)
    orders = np.zeros((rows, dimensions))
    points_store = []
    weights_store = []
    factor = 1
    

    # Now get the tensor grid for each sparse_index
    for i in range(0, rows):

        # loop through the dimensions
        for j in range(0, dimensions):
            orders[i,j] = np.array(sparse_index[i][j]) 

        # points and weights for each order~
        tensor = IndexSet('Tensor grid', orders[i,:])
        p2obj = Polyint(stackOfParameters, tensor)
        points, weights = p2obj.getPointsAndWeights(orders[i,:])
        del p2obj

        # Multiply weights by constant 'a':
        weights = weights * sparse_coeffs[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )
    
    # For normalizing!
    for i in range(0, dimensions):
        if flags[i] == 0:
            weights_store  = weights_store
        elif flags[i] == 1:
            weights_store = weights_store * (stackOfParameters[i].upper - stackOfParameters[i].lower )
            weights_store = weights_store/(2.0)

    # Now if the function is a callable, then we can compute the integral:
    if function is not None and callable(function):
        sparse_int = np.mat(weights_store) * evalfunction(points_store, function)
        point_store = removeDuplicates(points_store)
        return sparse_int, points_store
    else:
        point_store = removeDuplicates(points_store)
        return points_store, weights_store

def getPseudospectralCoefficients(self, function, override_orders=None):
    
    stackOfParameters = self.uq_parameters
    dimensions = len(stackOfParameters)
    q0 = [1.0]
    Q = []
    orders = []

    # If additional orders are provided, then use those!
    if override_orders is None:
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order)
            Qmatrix = stackOfParameters[i].getJacobiEigenvectors()
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])   
            
    else:
        for i in range(0, dimensions):
            orders.append(override_orders[i])
            Qmatrix = stackOfParameters[i].getJacobiEigenvectors(orders[i])
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    # Compute multivariate Gauss points and weights!
    if override_orders is None:
        p, w = self.getPointsAndWeights()
    else:
        p, w = self.getPointsAndWeights(override_orders)

    # Evaluate the first point to get the size of the system
    fun_value_first_point = function(p[0,:])
    u0 =  q0[0,0] * fun_value_first_point
    N = 1
    gn = int(np.prod(orders))
    Uc = np.zeros((N, gn))
    Uc[0,1] = u0

    function_values = np.zeros((1,gn))
    for i in range(0, gn):
        function_values[0,i] = function(p[i,:])

    # Now we evaluate the solution at all the points
    for j in range(0, gn): # 0
        Uc[0,j]  = q0[0,j] * function_values[0,j]

    # Compute the corresponding tensor grid index set:
    order_correction = []
    for i in range(0, len(orders)):
        temp = orders[i] - 1
        order_correction.append(temp)

    tensor_grid_basis = IndexSet('Tensor grid',  order_correction)
    tensor_set = tensor_grid_basis.getIndexSet()

    # Now we use kronmult
    K = efficient_kron_mult(Q, Uc)
    F = function_values
    K = np.column_stack(K)
    return K, tensor_set, p


def getSparsePseudospectralCoefficients(self, function):
    
    # INPUTS
    stackOfParameters = self.uq_parameters
    indexSets = self.index_sets
    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
    rows = len(sparse_indices)
    cols = len(sparse_indices[0])

    # For storage we use dictionaries
    individual_tensor_coefficients = {}
    individual_tensor_indices = {}
    points_store = {}
    indices = np.zeros((rows))


    for i in range(0,rows):
        orders = sparse_indices[i,:]
        K, I, points = getPseudospectralCoefficients(self, function, orders)
        individual_tensor_indices[i] = I
        individual_tensor_coefficients[i] =  K
        points_store[i] = points
        indices[i] = len(I)

    sum_indices = int(np.sum(indices))
    store = np.zeros((sum_indices, dimensions+1))
    points_saved = np.zeros((sum_indices, dimensions))
    counter = int(0)
    for i in range(0,rows):
        for j in range(0, int(indices[i])):
             store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][j]
             for d in range(0, dimensions):
                 store[counter,d+1] = individual_tensor_indices[i][j][d]
                 points_saved[counter,d] = points_store[i][j][d]
             counter = counter + 1

    # Now we use a while loop to iteratively delete the repeated elements while summing up the
    # coefficients!
    index_to_pick = 0
    flag = 1
    counter = 0

    rows = len(store)

    final_store = np.zeros((sum_indices, dimensions + 1))
    while(flag != 0):

        # find the repeated indices
        rep = find_repeated_elements(index_to_pick, store)
        coefficient_value = 0.0

        # Sum up all the coefficient values
        for i in range(0, len(rep)):
            actual_index = rep[i]
            coefficient_value = coefficient_value + store[actual_index,0]

        # Store into a new array
        final_store[counter,0] = coefficient_value
        final_store[counter,1::] = store[index_to_pick, 1::]
        counter = counter + 1

        # Delete index from store
        store = np.delete(store, rep, axis=0)

        # How many entries remain in store?
        rows = len(store)
        if rows == 0:
            flag = 0

    indices_to_delete = np.arange(counter, sum_indices, 1)
    final_store = np.delete(final_store, indices_to_delete, axis=0)

    # Now split final store into coefficients and their index sets!
    coefficients = np.zeros((1, len(final_store)))
    for i in range(0, len(final_store)):
        coefficients[0,i] = final_store[i,0]

    # Splitting final_store to get the indices!
    indices = final_store[:,1::]

    # Now just double check to make sure they are all integers
    for i in range(0, len(indices)):
        for j in range(0, dimensions):
            indices[i,j] = int(indices[i,j])
    
    K = np.column_stack(coefficients)
    return K, indices, points_saved

# Efficient kronecker product multiplication
# Adapted from David Gelich and Paul Constantine's kronmult.m
def efficient_kron_mult(Q, Uc):
    N = len(Q)
    n = np.zeros((N,1))
    nright = 1
    nleft = 1
    for i in range(0,N-1):
        rows_of_Q = len(Q[i])
        n[i,0] = rows_of_Q
        nleft = nleft * n[i,0]

    nleft = int(nleft)
    n[N-1,0] = len(Q[N-1]) # rows of Q[N]

    for i in range(N-1, -1, -1):
        base = 0
        jump = n[i,0] * nright
        for k in range(0, nleft):
            for j in range(0, nright):
                index1 = base + j
                index2 = int( base + j + nright * (n[i] - 1) )
                indices_required = np.arange(int( index1 ), int( index2 + 1 ), int( nright ) )
                small_Uc = np.mat(Uc[:, indices_required])
                temp = np.dot(Q[i] , small_Uc.T )
                temp_transpose = temp.T
                Uc[:, indices_required] = temp_transpose
            base = base + jump
        temp_val = np.max([i, 0]) - 1
        nleft = int(nleft/(1.0 * n[temp_val,0] ) )
        nright = int(nright * n[i,0])

    return Uc

# Routine for computing n choose k
def nchoosek(n, k):
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n - k)
    return (1.0 * numerator) / (1.0 * denominator)


