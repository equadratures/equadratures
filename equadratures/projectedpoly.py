"""The polynomial parent class."""
from .stats import Statistics
from .parameter import Parameter
from .basis import Basis
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
    def getZonotopeLinearInequalities(self):
        """
        Function that returns the linear inequalities that describe the zonotope.

        """
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
    def getQuadratureRule(self):
        """
        Returns a quadrature rule defined over the zonotope. 
        """