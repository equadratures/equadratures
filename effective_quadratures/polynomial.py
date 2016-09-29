"""Utilities for exploiting active subspaces when optimizing."""
from parameter import Parameter
from indexset import IndexSet
import numpy as np
from utils import error_function, evalfunction, find_repeated_elements

class Polynomial(object):
    
    """
    This subclass is an domains.ActiveVariableMap specifically for optimization.

    **See Also**

    optimizers.BoundedMinVariableMap
    optimizers.UnboundedMinVariableMap

    **Notes**

    This class's train function fits a global quadratic surrogate model to the
    n+2 active variables---two more than the dimension of the active subspace.
    This quadratic surrogate is used to map points in the space of active
    variables back to the simulation parameter space for minimization.
    """


    # Constructor
    def __init__(self, uq_parameters, method, index_sets=None):
        """
        Train the global quadratic for the regularization.

        :param ndarray X: input points used to train a global quadratic used in
            the `regularize_z` function.
        :param ndarray f: simulation outputs used to train a global quadratic in
            the `regularize_z` function.
        """
        self.uq_parameters = uq_parameters
        self.method = method

        # Here we set the index sets!
        if index_sets is None:

            # Determine the highest orders for a tensor grid
            highest_orders = []
            for i in range(0, len(uq_parameters)):
                highest_orders.append(uq_parameters[i].order)

            if(method == "tensor grid" or method == "Tensor grid"):
                indexObject = IndexSet(method, highest_orders)
                self.index_sets = indexObject
        else:

            if(method == "sparse grid" or method == "Sparse grid" or method == "spam" or method == "SPAM"):
                self.index_sets = index_sets

    # get methods
    def getCoefficients(self, function):
        """
        Train the global quadratic for the regularization.

        :param ndarray Y: N-by-n matrix of points in the space of active
            variables.
        :param int N: merely there satisfy the interface of `regularize_z`. It
            should not be anything other than 1.

        :return: Z, N-by-(m-n)-by-1 matrix that contains a value of the inactive
            variables for each value of the inactive variables.
        :rtype: ndarray

        **Notes**

        In contrast to the `regularize_z` in BoundedActiveVariableMap and
        UnboundedActiveVariableMap, this implementation of `regularize_z` uses
        a quadratic program to find a single value of the inactive variables
        for each value of the active variables.
        """
        if self.method == "tensor grid" or self.method == "Tensor grid":
            return getPseudospectralCoefficients(self.uq_parameters, function)
        if self.method == "spam" or self.method == "Spam":
            return getSparsePseudospectralCoefficients(self, function)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            print('WARNING: Use spam as a method instead!')
            return getSparseCoefficientsViaIntegration(self, function)

    def getPointsAndWeights(self, overwrite_orders=None):
        if overwrite_orders is not None:
            if self.method == "tensor grid" or self.method == "Tensor grid":
                return getGaussianQuadrature(self.uq_parameters, overwrite_orders)

            elif self.method == "sparse grid" or self.method == "Sparse grid" or self.method == "spam":
                p, w, sets = sparseGrid(self.uq_parameters, overwrite_orders)
                return p, w
        else:
            if self.method == "tensor grid" or self.method == "Tensor grid":
                return getGaussianQuadrature(self.uq_parameters)

            elif self.method == "sparse grid" or self.method == "Sparse grid" or self.method == "spam":
                p, w, sets = sparseGrid(self.uq_parameters, self.index_sets)
                return p, w

    def getPolynomialApproximation(self, function, plotting_pts):

        # Get the right polynomial coefficients
        if self.method == "tensor grid" or self.method == "Tensor grid":
            coefficients, indexset, evaled_pts = getPseudospectralCoefficients(self.uq_parameters, function)
        if self.method == "spam" or self.method == "Spam":
            coefficients, indexset, evaled_pts = getSparsePseudospectralCoefficients(self, function)
        if self.method == "sparse grid" or self.method == "Sparse grid":
            print('WARNING: Use spam as a method instead!')
            coefficients, indexset, evaled_pts = getSparseCoefficientsViaIntegration(self, function)

        P = getMultiOrthoPoly(self, plotting_pts, indexset)
        PolyApprox = np.mat(coefficients) * np.mat(P)
        return PolyApprox, evaled_pts

    def getMultivariatePolynomial(self, stackOfPoints, index_set_alternate=None):
        if index_set_alternate is None:
            index_set = self.indexsets
        else:
            index_set = index_set_alternate
        return getMultiOrthoPoly(self, stackOfPoints, index_set)

    
   # def getMultivariatePolynomialWithDerivatives(self, stackOfPoints, index_set_alternate=None):
   #     if index_set_alternate is None:
   #         index_set = self.indexsets
   #     else:
   #         index_set = index_set_alternate
   #     return getMultiOrthoPolyWithDerivative(self, stackOfPoints, index_set)


# Do not use the function below. It is provided here only for illustrative purposes.
# SPAM should be used!
def tensorGrid(listOfParameters, indexSet=None):

    # Get the tensor indices
    dimensions = len(listOfParameters)
    max_orders = []
    if indexSet is None:
        for u in range(0, dimensions):
            max_orders.append(int(listOfParameters[u].order) )
    else:
        max_orders = indexset.getMaxOrders()

    # Call the gaussian quadrature routine
    tensorObject = Polynomial(listOfParameters, method="tensor grid")
    points, weights = tensorGrid.getPointsAndWeights()

    return points, weights

def sparseGrid(listOfParameters, indexSet):

    # Get the number of parameters
    dimensions = len(listOfParameters)

    # Get the sparse index set attributes
    sparse_index, a , sg_set = IndexSet.getIndexSet(indexSet)
    rows = len(sparse_index)

    # Get this into an array
    orders = np.zeros((rows, dimensions))
    points_store = []
    weights_store = []
    factor = 1

    for i in range(0, rows):

        # loop through the dimensions
        for j in range(0, dimensions):
            orders[i,j] = np.array(sparse_index[i][j])

        # points and weights for each order~
        tensorObject = PolyParent(listOfParameters, method="tensor grid")
        points, weights = PolyParent.getPointsAndWeights(tensorObject, orders[i,:])

        # Multiply weights by constant 'a':
        weights = weights * a[i]

        # Now store point sets ---> scratch this, use append instead!!!!
        for k in range(0, len(points)):
            points_store = np.append(points_store, points[k,:], axis=0)
            weights_store = np.append(weights_store, weights[k])

    dims1 = int( len(points_store) / dimensions )
    points_store = np.reshape(points_store, ( dims1, dimensions ) )

    return points_store, weights_store, sg_set

# DO NOT USE!
def getSparseCoefficientsViaIntegration(self, function):

    # Preliminaries
    stackOfParameters = self.uq_parameters
    indexSets = self.index_sets
    dimensions = len(stackOfParameters)

    # Sparse grid integration rule
    pts, wts, sg_set_full = sparseGrid(stackOfParameters, indexSets)

    for i in range(0, len(sg_set_full)):
        for j in range(0, dimensions):
            sg_set_full[i,j] = int(sg_set_full[i,j])

    P = getMultiOrthoPoly(self, pts, sg_set_full)
    f = evalfunction(pts, function)
    f = np.mat(f)
    Wdiag = np.diag(wts)

    # Allocate memory for the coefficients
    rows = len(sg_set_full)
    coefficients = np.zeros((1, rows))

    # I multiply by P[0,:] because my zeroth order polynomial is not 1.0
    for i in range(0,rows):
        coefficients[0,i] = np.mat(P[i,:]) * Wdiag * np.diag(P[0,:]) * f

    return coefficients, sg_set_full, pts

# The SPAM technique!
def getSparsePseudospectralCoefficients(self, function):

    # INPUTS
    stackOfParameters = self.uq_parameters
    indexSets = self.index_sets
    dimensions = len(stackOfParameters)
    sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
    rows = len(sparse_indices)
    cols = len(sparse_indices[0])

    for i in range(0,rows):
        for j in range(0, cols):
            sparse_indices[i,j] = int(sparse_indices[i,j])

    # For storage we use dictionaries
    individual_tensor_coefficients = {}
    individual_tensor_indices = {}
    points_store = {}
    indices = np.zeros((rows,1))

    for i in range(0,rows):
        orders = sparse_indices[i,:]
        K, I, points = getPseudospectralCoefficients(self.uq_parameters, function, orders)
        individual_tensor_indices[i] = I
        individual_tensor_coefficients[i] =  K
        points_store[i] = points
        indices[i,0] = len(I)

    sum_indices = int(np.sum(indices))
    store = np.zeros((sum_indices, dimensions+1))
    points_saved = np.zeros((sum_indices, dimensions))
    counter = int(0)
    for i in range(0,rows):
        for j in range(0, int(indices[i][0])):
             store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][0][j]
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

    return coefficients, indices, points_saved

# Tensor grid pseudospectral method
def getPseudospectralCoefficients(stackOfParameters, function, additional_orders=None):

    dimensions = len(stackOfParameters)
    q0 = [1]
    Q = []
    orders = []

    # If additional orders are provided, then use those!
    if additional_orders is None:
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order)
            Qmatrix = stackOfParameters[i].getJacobiEigenvectors()
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    else:
        print 'Using custom coefficients!'
        for i in range(0, dimensions):
            orders.append(additional_orders[i])
            Qmatrix = stackOfParameters[i].getJacobiEigenvectors(orders[i])
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    # Compute multivariate Gauss points and weights
    p, w = getGaussianQuadrature(stackOfParameters, orders)

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

    tensor_grid_basis = IndexSet("tensor grid",  order_correction)
    tensor_set = tensor_grid_basis.getIndexSet()


    # Now we use kronmult
    K = efficient_kron_mult(Q, Uc)
    F = function_values
    K = np.column_stack(K)
    return K, tensor_set, p

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

# Computes nD quadrature points and weights using a kronecker product
def getGaussianQuadrature(stackOfParameters, additional_orders=None):

    # Initialize some temporary variables
    dimensions = int(len(stackOfParameters))
    orders = []

    # Check for extra input argument!
    if additional_orders is None:
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order)
    else:
        for i in range(0, dimensions):
            orders.append(additional_orders[i])

    # Initialize points and weights
    pp = [1.0]
    ww = [1.0]

     # number of parameters
    # For loop across each dimension
    for u in range(0,dimensions):

        # Call to get local quadrature method (for dimension 'u')
        local_points, local_weights = stackOfParameters[u].getLocalQuadrature(orders[u])

        # Tensor product of the weights
        ww = np.kron(ww, local_weights)

        # Tensor product of the points
        dummy_vec = np.ones((len(local_points), 1))
        dummy_vec2 = np.ones((len(pp), 1))
        left_side = np.array(np.kron(pp, dummy_vec))
        right_side = np.array( np.kron(dummy_vec2, local_points) )
        pp = np.concatenate((left_side, right_side), axis = 1)

    # Ignore the first column of pp
    points = pp[:,1::]
    weights = ww

    # Now re-scale the points and return only if its not a Gaussian!
    for i in range(0, dimensions):
        for j in range(0, len(points)):
            if (stackOfParameters[i].param_type == "Uniform"):
                #points[j,i] = points[j,i] * ( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound
                points[j,i] = 0.5 * ( points[j,i] + 1.0 )*( stackOfParameters[i].upper - stackOfParameters[i].lower) + stackOfParameters[i].lower

            elif (stackOfParameters[i].param_type == "Beta" ):
                points[j,i] =  ( points[j,i] )*( stackOfParameters[i].upper - stackOfParameters[i].lower) + stackOfParameters[i].lower

            elif (stackOfParameters[i].param_type == "Gaussian"):
                points[j,i] = points[j,i] # No scaling!

    # Return tensor grid quad-points and weights
    return points, weights

# determines a multivariate orthogonal polynomial corresponding to the stackOfParameters,
# their corresponding orders and then evaluates the polynomial at the corresponding
# stackOfPoints.
def getMultiOrthoPoly(self, stackOfPoints, index_set):

    # "Unpack" parameters from "self"
    stackOfParameters = self.uq_parameters
    dimensions = len(stackOfParameters)
    p = {}

    # Save time by returning if univariate!
    if(dimensions == 1):
        # Here "V" is the derivative. Need to change if we want to use multivariate
        # derivative polynomial.
        poly , V =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            p[i] = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )

    # Now we multiply components according to the index set
    no_of_points = len(stackOfPoints)
    polynomial = np.zeros((len(index_set), no_of_points))
    for i in range(0, len(index_set)):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][0][int(index_set[i,k])] * temp
            temp = polynomial[i,:]

    return polynomial

# Multivariate orthogonal polynomial with derivatives!
def getMultiOrthoPolyWithDerivative(self, stackOfPoints, index_set):
    
    # "Unpack" parameters from "self"
    stackOfParameters = self.uq_parameters
    dimensions = len(stackOfParameters)
    p = {}
    d = {}

    # Save time by returning if univariate!
    if(dimensions == 1):
        poly , derivatives =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
        return poly, derivatives
    else:
        for i in range(0, dimensions):
            poly, derivatives = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )
            p[i] = poly
            d[i] = derivatives

    # Now we multiply components according to the index set
    no_of_points = len(stackOfPoints)
    polynomial = np.zeros((len(index_set), no_of_points))
    derivatives = np.zeros((len(index_set), no_of_points, dimensions))

    for i in range(0, len(index_set)):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][0][int(index_set[i,k])] * temp
            temp = polynomial[i,:]
            derivatives[i,:,k] = d[k][0][int(index_set(i,k))] * temp
    
    return polynomial, derivatives
