"""The polynomial parent class."""
from .stats import Statistics
import pickle
from .parameter import Parameter
from .basis import Basis
import numpy as np
VERSION_NUMBER = 8.0

class Poly(object):
    """
    The class defines a Poly object. It is the parent class to Polyreg, Polyint and Polycs; the only difference between its children are the way in which the coefficients are computed. This class is defined by a list of Parameter objects and a Basis.

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.

    """
    def __init__(self, parameters, basis):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(parameters)
        self.orders = []
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        if not self.basis.orders :
            self.basis.setOrders(self.orders)
    def __setFunctionEvaluations__(self, function_evaluations):
        """
        Sets the function evaluations for the polynomial. This function can be called by the children of Poly.

        """
        self.function_evaluations = function_evaluations
    def __setCoefficients__(self, coefficients):
        """
        Sets the coefficients for polynomial. This function will be called by the children of Poly.

        :param Poly self:
            An instance of the Poly class.
        :param array coefficients:
            An array of the coefficients computed using either integration, least squares or compressive sensing routines.

        """
        self.coefficients = np.array(coefficients)
    def __setBasis__(self, basisNew):
        """
        Sets the basis
        """
        self.basis = basisNew 
    def __setQuadrature__(self, quadraturePoints, quadratureWeights):
        """
        Sets the quadrature points and weights

        :param Poly self:
            An instance of the Poly class.
        :param matrix quadraturePoints:
            A numpy matrix filled with the quadrature points.
        :param matrix quadratureWeights:
            A numpy matrix filled with the quadrature weights.
        """
        self.quadraturePoints = quadraturePoints
        self.quadratureWeights = quadratureWeights
    def __setDesignMatrix__(self, designMatrix):
        """
        Sets the design matrix assocaited with the quadrature (depending on the technique) points and the polynomial basis.

        :param Poly self:
            An instance of the Poly class.
        :param matrix designMatrix:
            A numpy matrix filled with the multivariate polynomial evaluated at the quadrature points.

        """
        self.designMatrix = designMatrix
    def clone(self):
        """
        Clones a Poly object.

        :param Poly self:
            An instance of the Poly class.
        :return:
            A clone of the Poly object.
        """
        return type(self)(self.parameters, self.basis)
    def getPolynomial(self, stackOfPoints, customBases=None):
        """
        Evaluates the value of each polynomial basis function at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the multivariate (in d-dimensions) polynomial basis functions must be evaluated.
        :return:
            A P-by-N matrix of polynomial basis function evaluations at the stackOfPoints, where P is the cardinality of the basis.
        """
        if customBases is None:
            basis = self.basis.elements
        else:
            basis = customBases
        basis_entries, dimensions = basis.shape

        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, __ = stackOfPoints.shape
        p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , _ , _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        polynomial = np.ones((basis_entries, no_of_points))
        for k in range(dimensions):
            basis_entries_this_dim = basis[:, k].astype(int)
            polynomial *= p[k][basis_entries_this_dim]

        return polynomial
    def getPolynomialGradient(self, stackOfPoints, dim_index = None):
        """
        Evaluates the gradient for each of the polynomial basis functions at a set of points,
        with respect to each input variable.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if len(stackOfPoints.shape) == 1:
            stackOfPoints = np.reshape(stackOfPoints, (len(stackOfPoints),1))
        no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _ , dpoly, _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis) ) )
            return dpoly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , dp[i], _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        R = []
        if dim_index is None:
            dim_index = range(dimensions)
        for v in range(dimensions):
            if not(v in dim_index):
                R.append(np.zeros((basis_entries, no_of_points)))
            else:
                polynomialgradient = np.ones((basis_entries, no_of_points))
                for k in range(dimensions):
                    basis_entries_this_dim = basis[:,k].astype(int)
                    if k==v:
                        polynomialgradient *= dp[k][basis_entries_this_dim]
                    else:
                        polynomialgradient *= p[k][basis_entries_this_dim]
                R.append(polynomialgradient)

        return R

    def getPolynomialHessian(self, stackOfPoints):
        """
        Evaluates the hessian for each of the polynomial basis functions at a set of points,
        with respect to each input variable.
        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d^2 elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}
        d2p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _, _, d2poly = self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
            return d2poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i], dp[i], d2p[i] = self.parameters[i]._getOrthoPoly(stackOfPoints[:, i],
                                                                       int(np.max(basis[:, i]) + 1))
        H = []
        for w in range(0, dimensions):
            gradDirection1 = w
            for v in range(0, dimensions):
                gradDirection2 = v
                polynomialhessian = np.zeros((basis_entries, no_of_points))
                for i in range(0, basis_entries):
                    temp = np.ones((1, no_of_points))
                    for k in range(0, dimensions):
                        if k == gradDirection1 == gradDirection2:
                            polynomialhessian[i, :] = d2p[k][int(basis[i, k])] * temp
                        elif k == gradDirection1:
                            polynomialhessian[i, :] = dp[k][int(basis[i, k])] * temp
                        elif k == gradDirection2:
                            polynomialhessian[i, :] = dp[k][int(basis[i, k])] * temp
                        else:
                            polynomialhessian[i, :] = p[k][int(basis[i, k])] * temp
                        temp = polynomialhessian[i, :]
                H.append(polynomialhessian)

        return H

    def getTensorQuadratureRule(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.

        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.basis.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._getLocalQuadrature(orders[u])
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
    def getStatistics(self, light=None, max_sobol_order=None):
        """
        Creates an instance of the Statistics class.

        :param Poly self:
            An instance of the Poly class.
        :param string quadratureRule:
            Two options exist for this string. The user can use 'qmc' for a distribution specific Monte Carlo (QMC) or they can use 'tensor grid' for standard tensor product grid. Typically, if the number of dimensions is less than 8, the tensor grid is the default option selected.
        :return:
            A Statistics object.
        """
        if light is None:
            evals = self.getPolynomial(self.quadraturePoints)
            return Statistics(self.coefficients, self.basis, self.parameters, self.quadraturePoints, self.quadratureWeights, evals, max_sobol_order)
        else:
            return Statistics(self.coefficients, self.basis, self.parameters, max_sobol_order=max_sobol_order)            
    def getQuadratureRule(self, options=None, number_of_points = None):
        """
        Generates quadrature points and weights.

        :param Poly self:
            An instance of the Poly class.
        :param string options:
            Two options exist for this string. The user can use 'qmc' for a distribution specific Monte Carlo (QMC) or they can use 'tensor grid' for standard tensor product grid. Typically, if the number of dimensions is less than 8, the tensor grid is the default option selected.
        :param int number_of_points:
            If QMC is chosen, specifies the number of quadrature points in each direction. Otherwise, this is ignored.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        if options is None:
            if self.dimensions > 5 or np.max(self.orders) > 4:
                options = 'qmc'
            else:
                options = 'tensor grid'
        if options.lower() == 'qmc':
            if number_of_points is None:
                default_number_of_points = 20000
            else:
                default_number_of_points = number_of_points
            p = np.zeros((default_number_of_points, self.dimensions))
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            for i in range(0, self.dimensions):
                p[:,i] = np.array(self.parameters[i].getSamples(default_number_of_points)).reshape((default_number_of_points,))
            return p, w

        if options.lower() == 'tensor grid' or options.lower() == 'quadrature':
            p,w = self.getTensorQuadratureRule([i for i in self.basis.orders])
            return p,w
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
        if self.dimensions == 1:
            return np.mat(self.coefficients).T * H
        hess = np.zeros((self.dimensions, self.dimensions, no_of_points))
        for i in range(0, self.dimensions):
            for j in range(0, self.dimensions):
                hess[i, j, :] = np.mat(self.coefficients).T * H[i * self.dimensions + j]
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
        return lambda x : self.evaluatePolyHessFit(x)
    def getFunctionSamples(self, number_of_samples):
        """
        Returns a set of function samples; useful for computing probabilities.

        :param Poly self:
            An instance of the Poly class.
        :param callable function:
            A callable function (or evaluations of the function at the prerequisite quadrature points).
        :param array coefficients:
            A numpy array of the coefficients
        :param matrix indexset:
            A K-by-d matrix of the index set.
        :return:
            A 50000-by-1 array of function evaluations.

        """
        dimensions = self.dimensions
        if number_of_samples is None:
            number_of_samples = 50000 # default value!
        plotting_pts = np.zeros((number_of_samples, dimensions))
        for i in range(0, dimensions):
                univariate_samples = self.parameters[i].getSamples(number_of_samples)
                for j in range(0, number_of_samples):
                    plotting_pts[j, i] = univariate_samples[j]
        samples = self.evaluatePolyFit(plotting_pts)
        return plotting_pts, samples

    def savePoly(self, filename, full = False):
        """
        Saves the poly object to a pickle file (.pkl)
        :param filename: Filename to save to
        :param full: If False, only saves some attributes of the class to minimize file size
        :return:
        """
        poly_minimal = Poly_minimal(self)
        if full:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(poly_minimal,f)

    @staticmethod
    def convert2Full(poly_minimal):
        """
        Converts a Poly_minimal object to a full Poly object, running through the constructor.
        This gives some basic functionalities such as evaluating the polynomial fit and gradients.
        Note: This method is not compliant with the EQ v8.0 philosophy, and is just temporary.
        :param poly_minimal: A Poly_minimal object
        :return: poly object
        """
        basis = Basis(poly_minimal.basis_type, poly_minimal.basis_orders, poly_minimal.basis_level,
                      poly_minimal.basis_growth_rule, poly_minimal.basis_q)
        num_params = len(poly_minimal.orders)
        parameters = []
        for i in range(num_params):
            parameters.append(Parameter(poly_minimal.orders[i], poly_minimal.distributions[i],
                                        endpoints = poly_minimal.endpoints[i],
                                        shape_parameter_A = poly_minimal.shape_parameter_As[i],
                                        shape_parameter_B = poly_minimal.shape_parameter_Bs[i],
                                        lower = poly_minimal.lowers[i],
                                        upper = poly_minimal.uppers[i],
                                        data = poly_minimal.datas[i]))
        poly = Poly(parameters, basis)
        if hasattr(poly_minimal,'coefficients'):
            poly.__setCoefficients__(poly_minimal.coefficients)
        if hasattr(poly_minimal,'quadraturePoints'):
            poly.__setQuadrature__(poly_minimal.quadraturePoints, poly_minimal.quadratureWeights)
        return poly

    def convert2Min(self):
        """
        Converts self to a Poly_minimal instance
        :return: poly_min: A Poly_minimal object
        """
        return Poly_minimal(self)

class Poly_minimal(object):
    """Stores the bare minimum of a Poly object, which is to be reinstantiated upon loading."""
    def __init__(self, poly):
        #self, order, distribution, endpoints = False, shape_parameter_A = None, shape_parameter_B = None, lower = None, upper = None, data = None
        self.orders = []
        self.distributions = []
        self.endpoints = []
        self.shape_parameter_As = []
        self.shape_parameter_Bs = []
        self.lowers = []
        self.uppers = []
        self.datas = []
        for p in poly.parameters:
            self.orders.append(p.order)
            self.distributions.append(p.name)
            self.endpoints.append(p.endpoints)
            self.shape_parameter_As.append(p.shape_parameter_A)
            self.shape_parameter_Bs.append(p.shape_parameter_B)
            self.lowers.append(p.lower)
            self.uppers.append(p.upper)
            self.datas.append(p.data)

        #basis_type, orders = None, level = None, growth_rule = None, q = None
        self.basis_type = poly.basis.basis_type
        self.basis_orders = poly.basis.orders
        self.basis_level = poly.basis.level
        self.basis_growth_rule = poly.basis.growth_rule
        self.basis_q = poly.basis.q
        # self.basis = poly.basis

        try:
            self.coefficients = poly.coefficients
        except AttributeError:
            pass
        try:
            self.quadraturePoints = poly.quadraturePoints
            self.quadratureWeights = poly.quadratureWeights
        except AttributeError:
            pass
        # try:
        #     self.designMatrix = poly.designMatrix
        # except AttributeError:
        #     pass
