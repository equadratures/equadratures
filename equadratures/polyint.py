"""Operations involving multivariate polynomials (without gradients) via numerical quadrature. The following quadrature techniques are available for coefficient computation:
    1. Tensor grids;
    2. Sparse pseudospectral approximation method;

References:
    - Constantine, P. G., Eldred, M. S., & Phipps, E. T. (2012). Sparse pseudospectral approximation method. Computer Methods in Applied Mechanics and Engineering, 229, 1-12. `Paper <https://www.sciencedirect.com/science/article/pii/S0045782512000953>`_.
"""
from parameter import Parameter
from basis import Basis
from basis import sparse_grid_basis
from utils import find_repeated_elements, evalfunction
from poly import Poly
import numpy as np

class Polyint(Poly):
    """
    This class defines a Polyint (polynomial via integration) object

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.
    :param string sampling:
        The sampling technique. Choose from: 'tensor grid quadrature' (default), 'sparse grid quadrature', 'effectively subsampled quadrature', 'Christoffel subsamples', 'Induced subsamples' and 'randomized quadrature'.
    :param callable fun:
        Instead of specifying the output training points, the user can also provide a callable function, which will be evaluated.
    """
    def __init__(self, parameters, basis):
        super(Polyint, self).__init__(parameters, basis)

    def computeCoefficients(self, function):
        """
        Returns multivariate orthonormal polynomial coefficients.

        :param Polyint self: An instance of the Polyint class
        :param: callable function: The function that needs to be approximated (or interpolated)
        :return: coefficients: The pseudospectral coefficients
        :rtype: ndarray
        :return: indexset: The indices used for the pseudospectral computation
        :rtype: ndarray
        :return: evaled_pts: The points at which the function was evaluated
        :rtype: ndarray

        """
        # Method to compute the coefficients
        method = self.basis.basis_type
        if method.lower() == 'sparse grid':
            coefficients, indexset, evaled_pts, weights = getSparsePseudospectralCoefficients(self, function)
        elif (method.lower() == 'tensor grid') or (method.lower() == 'tensor'):
            coefficients, indexset, evaled_pts, weights = getPseudospectralCoefficients(self, function)
            self.basis.elements = indexset
        self.coefficients = coefficients
        self.multi_index = indexset
        self.quadraturePoints = evaled_pts
        self.quadratureWeights = weights
        super(Polyint, self).__setCoefficients__(self.coefficients)


#--------------------------------------------------------------------------------------------------------------
#
#  PRIVATE FUNCTIONS!
#
#--------------------------------------------------------------------------------------------------------------
def getPseudospectralCoefficients(self, function, override_orders=None):
    if override_orders is None:
        pts, wts = super(Polyint, self).getTensorQuadratureRule()
    else:
        pts, wts = super(Polyint, self).getTensorQuadratureRule(override_orders)
    m = len(wts)
    P = super(Polyint, self).getPolynomial(pts)
    W = np.mat( np.diag(np.sqrt(wts)))
    A = np.mat(W * P.T)
    if callable(function):
        y = evalfunction(points=pts, function=function)
    else:
        y = function
    b = np.dot( W  ,  np.reshape(y, (m,1)) )
    coefficients = np.dot(A.T , b)  
    return coefficients, self.basis.elements, pts, wts
    
    """
    stackOfParameters = self.parameters
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
            Qmatrix = stackOfParameters[i].getJacobiEigenvectors(orders[i]+1)
            Q.append(Qmatrix)

            if orders[i] + 1 == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    # Compute multivariate Gauss points and weights!
    if override_orders is None:
        p, w = self.getTensorQuadratureRule()
    else:
        p, w = self.getTensorQuadratureRule(override_orders)

    # Evaluate the first point to get the size of the system
    fun_value_first_point = function(p[0,:])
    u0 =  q0[0,0] * fun_value_first_point
    N = 1
    orders_plus_one = [x+1 for x in orders]
    gn = int(np.prod(orders_plus_one))
    Uc = np.zeros((N, gn))
    Uc[0,0] = u0
    function_values = np.zeros((1,gn))
    for i in range(0, gn):
        function_values[0,i] = function(p[i,:])

    # Now we evaluate the solution at all the points
    for j in range(1, gn): # 0
        Uc[0,j]  = q0[0,j] * function_values[0,j]
    basis = Basis('Tensor grid',  orders)
    tensor_set = basis.elements

    values = np.sum(tensor_set, 1)
    g = np.argsort(values)
    new_basis = tensor_set[g,:]

    # Now we use kronmult
    K = efficient_kron_mult(Q, Uc)
    K[0,:] = K[0,g]
    F = function_values
    K = np.column_stack(K)
    """

def getSparsePseudospectralCoefficients(self, function):

    # INPUTS
    stackOfParameters = self.parameters
    indexSets = self.basis
    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, not_used = sparse_grid_basis(self.basis.level, self.basis.growth_rule, self.dimensions)
    rows = len(sparse_indices)
    cols = len(sparse_indices[0])

    # For storage we use dictionaries
    individual_tensor_coefficients = {}
    individual_tensor_indices = {}
    points_store = {}
    weights_store = {}
    indices = np.zeros((rows))

    for i in range(0,rows):
        orders = sparse_indices[i,:] 
        K, I, points , weights = getPseudospectralCoefficients(self, function, orders)
        individual_tensor_indices[i] = I
        individual_tensor_coefficients[i] =  K
        points_store[i] = points
        weights_store[i] = weights
        indices[i] = len(I)

    sum_indices = int(np.sum(indices))
    store = np.zeros((sum_indices, dimensions+1))
    points_saved = np.zeros((sum_indices, dimensions))
    weights_saved = np.zeros((sum_indices))
    counter = int(0)
    for i in range(0,rows):
        for j in range(0, int(indices[i])):
             store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][j]
             for d in range(0, dimensions):
                 store[counter,d+1] = individual_tensor_indices[i][j][d]
                 points_saved[counter,d] = points_store[i][j][d]
             weights_saved[counter] = weights_store[i][j]
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
    return K, indices, points_saved, weights_saved
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
def nchoosek(n, k):
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n - k)
    return (1.0 * numerator) / (1.0 * denominator)
