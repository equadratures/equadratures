"""The polynomial parent class."""
from .stats import Statistics
from .parameter import Parameter
from .basis import Basis
from scipy.spatial import ConvexHull
import numpy as np
VERSION_NUMBER = 7.6

class Projectedpoly(object):
    """
    The class defines a Projectedpoly object.

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.

    """
    def __init__(self, parameters, basis, subspace):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(parameters)
        self.subspace = subspace
        rows, cols = self.subspace.shape
        if rows > cols:
            self.reduced_dimensions = cols 
        else:
            self.reduced_dimensions = rows
            self.subspace = self.subspace.T
        self.orders = []
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        if not self.basis.orders :
            self.basis.setOrders(self.orders)
    def getIntervalVertices(self):
        W = self.subspace
        d, n = W.shape
        assert n == 1
        y0 = np.dot(W.T, np.sign(W))[0]
        if y0 < -y0:
            yl, yu = y0, -y0
            xl, xu = np.sign(W), -np.sign(W)
        else:
            yl, yu = -y0, y0
            xl, xu = -np.sign(W), np.sign(W)
        Y = np.array([yl, yu]).reshape((2,1))
        X = np.vstack((xl.reshape((1,d)), xu.reshape((1,d))))
        return Y, X
    def getZonotopeVertices(self):
        """
        Function that returns the vertices that describe the zonotope.

        """
        return Y, X
    def getZonotopeLinearInequalities(self):
        """
        Function that returns the linear inequalities that describe the zonotope.

        """
        W = self.subspace
        d, n = W.shape
        if n == 1:
            Y, _ = self.getIntervalVertices()
            pass
# NOTE:     What to return if ridge function is 1d???
        else:
            Y, _ = self.getZonotopeVertices()
            convexHull = ConvexHull(Y)
            A = convexHull.equations[:,:n]
            b = convexHull.equations[:,n]
            return A, b
    def approxFullSpacePolynomial(self):
        """
        Use the quadratic program to approximate the polynomial over the full space.
        """
        Polyfull = Poly()
        return Polyfull
    def getPolynomial(self, stackOfPoints):
        """
        Evaluates the value of each polynomial basis function at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the multivariate (in d-dimensions) polynomial basis functions must be evaluated.
        :return:
            A P-by-N matrix of polynomial basis function evaluations at the stackOfPoints, where P is the cardinality of the basis.

        """
# NOTE:     Should this be w.r.t. x or u?    
    def getPolynomialGradient(self, stackOfPoints, dim_index = None):
        """
        Evaluates the gradient for each of the polynomial basis functions at a set of points,
        with respect to the project of the input variable.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
# NOTE:     Should this be w.r.t. x or u?    
    def getPolynomialHessian(self, stackOfPoints):
        """
        Evaluates the hessian for each of the polynomial basis functions at a set of points,
        with respect to the project of the input variable.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d^2 elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
# NOTE:     Should this be w.r.t. x or u?    
    def getQuadratureRule(self):
        """
        Returns a quadrature rule defined over the zonotope. 
        """
# NOTE:     Not entirely sure what I should do for this...
    def evaluatePolyFit(self, stackOfPoints):
        """
        Evaluates the the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A 1-by-N matrix of the polynomial approximation.
        """
        return self.getPolynomial(stackOfPoints).T *  np.mat(self.coefficients)
    def evaluatePolyGradFit(self, stackOfPoints, dim_index = None):
        """
        Evaluates the gradient of the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A d-by-N matrix of the gradients of the polynomial approximation.

        **Notes:**

        This function should not be confused with getPolynomialGradient(). The latter is only concerned with approximating what the multivariate polynomials
        gradient values are at prescribed points.
        """
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        H = self.getPolynomialGradient(stackOfPoints, dim_index=dim_index)
        grads = np.zeros((self.dimensions, no_of_points ) )
        if self.dimensions == 1:
            return np.mat(self.coefficients).T * H
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads
    def evaluatePolyHessFit(self, stackOfPoints):
        """
        Evaluates the hessian of the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A d-by-d-by-N matrix of the hessian of the polynomial approximation.
        """
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        H = self.getPolynomialHessian(stackOfPoints)
        hess = np.zeros( (self.dimensions, self.dimensions,no_of_points) )
        for i in range(0, self.dimensions):
            for j in range(0, self.dimensions):
                hess[i,j,:] = np.mat(self.coefficients).T * H[i * self.dimensions + j]
        return hess
    def getPolyFitFunction(self):
        """
        Returns a callable polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.

        """
        return lambda x: np.array(self.getPolynomial(x).T *  np.mat(self.coefficients))
    def getPolyGradFitFunction(self):
        """
        Returns a callable for the gradients of the polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.

        """
        return lambda x : self.evaluatePolyGradFit(x)
    def getPolyHessFitFunction(self):
        """
        Returns a callable for the hessian of the polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.

        """
        return lambda (x) : self.evaluatePolyHessFit(x)