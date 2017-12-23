"""Operations involving polynomial regression on a data set"""
from parameter import Parameter
from basis import Basis
import numpy as np
from stats import Statistics, getAllSobol
import scipy


class Polyreg(object):
    """
    This class defines a Polyreg (polynomial via regression) object
    :param training_x: A numpy
    :param IndexSet basis: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no basis parameter input is given.
    :param parameters: List of instances of Parameters class.
    :param training_y: Column vector (np array) of regression targets corresponding to each row of training_x. Either this or fun should be specified, but not both.
    :param fun: Function to evaluate training_x on to obtain regression targets automatically. Either this or fun should be specified, but not both.

    """
    # Constructor
    def __init__(self, training_x, parameters, basis, fun=None, training_y=None):
        self.x = training_x
        assert self.x.shape[1] == len(parameters) # Check that x is in the correct shape
        if not((training_y is None) ^ (fun is None)):
            raise ValueError("Specify only one of fun or training_y.")
        if not(fun is None):
            try:
                self.y = np.apply_along_axis(fun, 1, self.x)
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = training_y
        self.basis = basis
        self.dimensions = len(parameters)
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polyreg:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
        self.parameters = parameters
        self.A =  getPolynomial(self.parameters, self.scalingX(self.x), self.basis).T
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1))
        self.computeCoefficients()

    def scalingX(self, x_points_scaled):
        rows, cols = x_points_scaled.shape
        points = np.zeros((rows, cols))
        points[:] = x_points_scaled

        # Now re-scale the points and return only if its not a Gaussian!
        for i in range(0, self.dimensions):
            for j in range(0, rows):
                if (self.parameters[i].param_type == "Uniform"):
                    points[j,i] = 2.0 * ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) - 1.0
                elif (self.parameters[i].param_type == "Beta" ):
                    points[j,i] =  ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower)
        return points

    # Solve for coefficients using ordinary least squares
    def computeCoefficients(self):
        alpha = np.linalg.lstsq(self.A, self.y) # Opted for numpy's standard version because of speed!
        self.coefficients = alpha[0]

    def getfitStatistics(self):
        t_stat = get_t_value(self.coefficients, self.A, self.y)
        r_sq = get_R_squared(self.coefficients, self.A, self.y)
        return t_stat, r_sq

    def getStatistics(self, quadratureRule=None):
        p, w = self.getQuadratureRule(quadratureRule)
        evals = getPolynomial(self.parameters, p, self.basis)
        return Statistics(self.coefficients, self.basis, self.parameters, p, w, evals)

    def getPolynomialApproximant(self):
        return self.A * np.mat(self.coefficients)

    def getPolynomialGradientApproximant(self, direction=None, xvalue=None):
        if xvalue is None:
            xvalue = self.x

        if direction is not None:
            C = getPolynomialGradient(self.parameters, self.scalingX(xvalue), self.basis, direction).T
            return C * np.mat(self.coefficients)
        else:
            H = getPolynomialGradient(self.parameters, self.scalingX(xvalue), self.basis)
            grads = np.zeros((self.dimensions, len(xvalue) ) )
            for i in range(0, self.dimensions):
                grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads

    def getPolyFit(self):
        return lambda (x): getPolynomial(self.parameters, self.scalingX(x) , self.basis).T *  np.mat(self.coefficients)

    def getPolyGradFit(self):
        return lambda (x) : self.getPolynomialGradientApproximant(xvalue=x)

    def getQuadratureRule(self, options=None):
        if options is None:
            if self.dimensions > 5:
                options = 'qmc'
            elif self.dimensions < 5 :
                options = 'tensor grid'

        if options.lower() == 'qmc':
            default_number_of_points = 10000
            p = np.zeros((default_number_of_points, self.dimensions))
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            print self.dimensions
            for i in range(0, self.dimensions):
                p[:,i] = self.parameters[i].getSamples(m=default_number_of_points).reshape((default_number_of_points,))
            return p, w

        if options.lower() == 'tensor grid':
            return getTensorQuadratureRule(self.parameters, self.dimensions, self.basis.orders)

    def computeActiveSubspaces(self, samples=None):
        d = self.dimensions
        if samples is  None:
            M = 300
            X = np.zeros((M, d))
            for j in range(0, d):
                X[:, j] =  np.reshape( self.parameters[j].getSamples(M), M)
                print np.max(X[:,j]), np.min(X[:,j])
        else:
            X = samples
            M, _ = X.shape
            X = samples

        # Gradient matrix!
        polygrad = self.getPolynomialGradientApproximant(xvalue=X)
        weights = np.ones((M, 1)) / M

        R = polygrad.transpose() * weights
        C = np.dot(polygrad, R )
        #C = np.dot(polygrad.transpose(), np.dot( polygrad , weights) )
        #print C.shape
        #print polygrad.transpose().shape

        # Construct covariance matrix!
        #C = np.zeros((d, d))
        #for i in range(0, d):
        #        grad_f = np.mat( np.reshape(polygrad[:,i], (d, 1)) )
        #    C =+ grad_f * grad_f.T
        #C = 1./float(M) * C

        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]
        return eigs, eigVecs

def get_t_value(alpha, A, y):
    RSS = np.linalg.norm(y - np.dot(A,alpha))**2
    p, n = A.shape
    if n == p:
        return "exact"
    RSE = RSS/(n-p)
    Q, R = np.linalg.qr(self.A)
    inv_ATA = np.linalg.inv(np.dot(R.T, R))
    se = np.array([np.sqrt(RSE * inv_ATA[j,j]) for j in range(p)])
    t_stat = alpha / np.reshape(se, (len(se), 1))
    return t_stat

def getTensorQuadratureRule(stackOfParameters, dimensions, orders):
        flag = 0

        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        # number of parameters
        # For loop across each dimension
        for u in range(0, dimensions):

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

        # Return tensor grid quad-points and weights
        return points, weights



def get_R_squared(alpha, A, y):
    y_bar = scipy.mean(y) * np.ones(len(y))
    TSS = np.linalg.norm(y - y_bar)**2
    RSS = np.linalg.norm(np.dot(A,alpha) - y)**2
    return 1 - RSS/TSS

def getPolynomial(stackOfParameters, stackOfPoints, chosenBasis):
    # "Unpack" parameters from "self"
    basis = chosenBasis.elements
    basis_entries, dimensions = basis.shape
    no_of_points, _ = stackOfPoints.shape
    polynomial = np.zeros((basis_entries, no_of_points))
    p = {}

    # Save time by returning if univariate!
    if dimensions == 1:
        poly , _ =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            if len(stackOfPoints.shape) == 1:
                stackOfPoints = np.array([stackOfPoints])
            p[i] , _ = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i]) + 1 ) )

    # One loop for polynomials
    for i in range(0, basis_entries):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][int(basis[i,k])] * temp
            temp = polynomial[i,:]
    return polynomial

def getPolynomialGradient(stackOfParameters, stackOfPoints, chosenBasis):
     # "Unpack" parameters from "self"
    basis = chosenBasis.elements
    basis_entries, dimensions = basis.shape
    no_of_points, _ = stackOfPoints.shape
    p = {}
    dp = {}

    # Save time by returning if univariate!
    if dimensions == 1:
        poly , _ =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            if len(stackOfPoints.shape) == 1:
                stackOfPoints = np.array([stackOfPoints])
            p[i] , dp[i] = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i]) + 1 ) )

    # One loop for polynomials
    R = []
    for v in range(0, dimensions):
        gradDirection = v
        polynomialgradient = np.zeros((basis_entries, no_of_points))
        for i in range(0, basis_entries):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                if k == gradDirection:
                    polynomialgradient[i,:] = dp[k][int(basis[i,k])] * temp
                else:
                    polynomialgradient[i,:] = p[k][int(basis[i,k])] * temp
                temp = polynomialgradient[i,:]
        R.append(polynomialgradient)

    return R
