"""Polynomial Parent Class"""
import numpy as np
from parameter import Parameter
from basis import Basis
from stats import Statistics

class Poly(object):
    """
    The class defines a Poly object. It is the parent class to Polyreg, Polyint, Polylsq and Polycs. 
    It is defined by a list of Parameter objects and a Basis.
    """
    def __init__(self, parameters, basis):
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(parameters)
        self.orders = []
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        self.basis.setOrders(self.orders)

    def __setCoefficients__(self, coefficients):
        self.coefficients = coefficients

    def __setDesignMatrix__(self, designMatrix):
        self.designMatrix = designMatrix

    def clone(self):
        return type(self)(self.parameters, self.basis)

    def scaleInputs(self, x_points_scaled):
        rows, cols = x_points_scaled.shape
        points = np.zeros((rows, cols))
        points[:] = x_points_scaled

        # Now re-scale 
        for i in range(0, self.dimensions):
            for j in range(0, rows):
                if (self.parameters[i].param_type == "Uniform"):
                    points[j,i] = 2.0 * ( ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) ) - 1.0
                elif (self.parameters[i].param_type == "Beta" ):
                    points[j,i] =  ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) 
        return points        

    def getPolynomial(self, stackOfPoints):
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        no_of_points, _ = stackOfPoints.shape
        polynomial = np.zeros((basis_entries, no_of_points))
        p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        for i in range(0, basis_entries):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                polynomial[i,:] = p[k][int(basis[i,k])] * temp
                temp = polynomial[i,:]
        
        return polynomial

    def getPolynomialGradient(self, stackOfPoints):
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints)
            return poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , dp[i] = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i]) + 1 ) )

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

    def getTensorQuadratureRule(self, orders=None):
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._getLocalQuadrature(orders[u]+1, scale=True)
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

    def getStatistics(self, quadratureRule=None):
        p, w = self.getQuadratureRule(quadratureRule)
        evals = self.getPolynomial(self.scaleInputs(p))
        return Statistics(self.coefficients, self.basis, self.parameters, p, w, evals)

    def getQuadratureRule(self, options=None):
        if options is None:
            if self.dimensions > 8:
                options = 'qmc'
            elif self.dimensions < 8 :
                options = 'tensor grid'
        if options.lower() == 'qmc':
            default_number_of_points = 20000
            p = np.zeros((default_number_of_points, self.dimensions)) 
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            for i in range(0, self.dimensions):
                p[:,i] = self.parameters[i].getSamples(m=default_number_of_points).reshape((default_number_of_points,))
            return p, w
            
        if options.lower() == 'tensor grid':
            p,w = self.getTensorQuadratureRule([2*i for i in self.basis.orders])
            return p,w

    def evaluatePolyGradFit(self, xvalue):
        H = self.getPolynomialGradient(self.scaleInputs(xvalue))
        grads = np.zeros((self.dimensions, len(xvalue) ) )
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads

    def getPolyFitFunction(self, x):
        return lambda (x): self.getPolynomial(self.scaleInputs(x)).T *  np.mat(self.coefficients)
    
    def evaluatePolyFit(self, x):
        return self.getPolynomial(self.scaleInputs(x)).T *  np.mat(self.coefficients)
    
    def getPolyGradFitFunction(self):
        return lambda (x) : self.evaluatePolyGradFit(xvalue=x)


