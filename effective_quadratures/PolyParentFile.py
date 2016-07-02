#!/usr/bin/env python
from PolyParams import PolynomialParam
from IndexSets import IndexSet
import numpy as np
import Utils as utils
"""

    Polyparent Class

    Pranay Seshadri
    ps583@cam.ac.uk

"""
class PolyParent(object):
    """
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                constructor / initializer
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def __init__(self, uq_parameters, method, index_sets=None):

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

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    get() methods
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    def getMultivariatePoly(self, points):
        return getMultiOrthoPoly(self, points)

    def getCoefficients(self, function):
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
        else:
            if self.method == "tensor grid" or self.method == "Tensor grid":
                return getGaussianQuadrature(self.uq_parameters)

        if self.method == "sparse grid" or self.method == "Sparse grid":
            indexSets = self.index_sets
            level = self.level
            growth_rule = self.growth_rule
            sparse_indices, sparse_factors, not_used = IndexSet.getIndexSet(indexSets)
            return sparsegrid(self.uq_parameters, self.index_sets, level, growth_rule)

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

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE FUNCTIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
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
        max_orders = IndexSet.getMaxOrders(indexSet)

    # Call the gaussian quadrature routine
    tensorObject = PolyParent(listOfParameters, method="tensor grid")
    points, weights = PolyParent.getPointsAndWeights(tensorObject)

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

def getSparseCoefficientsViaIntegration(self, function):

    # Preliminaries
    stackOfParameters = self.uq_parameters
    indexSets = self.index_sets
    dimensions = len(stackOfParameters)

    # Sparse grid integration rule
    pts, wts, sg_set_full = sparseGrid(stackOfParameters, indexSets)
    print '~~~~~~~~Full Sparse Grid Index Set~~~~~~~~~~~'
    print sg_set_full

    for i in range(0, len(sg_set_full)):
        for j in range(0, dimensions):
            sg_set_full[i,j] = int(sg_set_full[i,j])

    print len(pts), len(wts), len(sg_set_full), len(sg_set_full[0,:])
    print sg_set_full
    P = getMultiOrthoPoly(self, pts, sg_set_full)
    print len(P), len(P[0,:])
    print '---------------------'
    f = utils.evalfunction(pts, function)
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
        rep = utils.find_repeated_elements(index_to_pick, store)
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
            orders.append(stackOfParameters[i].order )
            Qmatrix = PolynomialParam.getJacobiEigenvectors(stackOfParameters[i])
            Q.append(Qmatrix)

            if orders[i] == 1:
                q0 = np.kron(q0, Qmatrix)
            else:
                q0 = np.kron(q0, Qmatrix[0,:])

    else:
        for i in range(0, dimensions):
            orders.append(additional_orders[i])
            Qmatrix = PolynomialParam.getJacobiEigenvectors(stackOfParameters[i], orders[i])
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
    tensor_set = IndexSet.getIndexSet(tensor_grid_basis)


    # Now we use kronmult
    K = efficient_kron_mult(Q, Uc)
    F = function_values

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
        local_points, local_weights = PolynomialParam.getLocalQuadrature(stackOfParameters[u], orders[u])

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
            if (stackOfParameters[i].param_type != "Gaussian" and stackOfParameters[i].param_type != "Normal" and stackOfParameters[i].param_type != "Beta")  :
                points[j,i] = 0.5 * ( points[j,i] + 1.0 )*( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound

            elif (stackOfParameters[i].param_type == "Beta" ):
                points[j,i] =  ( points[j,i] )*( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound

            # Scale points by the mean!
            elif (stackOfParameters[i].param_type == "Gaussian" or stackOfParameters[i].param_type == "Normal" ):
                points[j,i] = points[j,i] + float(stackOfParameters[i].shape_parameter_A)

    # Return tensor grid quad-points and weights
    return points, weights

# determines a multivariate orthogonal polynomial corresponding to the stackOfParameters,
# their corresponding orders and then evaluates the polynomial at the corresponding
# stackOfPoints.
def getMultiOrthoPoly(self, stackOfPoints, index_set_alternate=None):

    # "Unpack" parameters from "self"
    stackOfParameters = self.uq_parameters

    # Now if the user has provided an alternate index set, we use that one.
    if index_set_alternate is None:
        index_set = self.indexsets
    else:
        index_set = index_set_alternate

    dimensions = len(stackOfParameters)
    p = {}

    # Save time by returning if univariate!
    if(dimensions == 1):
        # Here "V" is the derivative. Need to change if we want to use multivariate
        # derivative polynomial.
        poly , V =  PolynomialParam.getOrthoPoly(stackOfParameters[0], stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            p[i] = PolynomialParam.getOrthoPoly(stackOfParameters[i], stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )

    # Now we multiply components according to the index set
    no_of_points = len(stackOfPoints)
    polynomial = np.zeros((len(index_set), no_of_points))
    for i in range(0, len(index_set)):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][0][int(index_set[i,k])] * temp
            temp = polynomial[i,:]

    return polynomial
