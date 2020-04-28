""" Utilities for dealing with correlated inputs."""
from equadratures.parameter import Parameter
from equadratures.poly import Poly, evaluate_model, evaluate_model_gradients
from equadratures.basis import Basis
import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal, norm
from scipy.linalg import lu
from scipy import optimize
from copy import deepcopy

class Correlations(object):
    """
    The class defines a Nataf transformation. The input correlated marginals are mapped from their physical space to a new
    standard normal space, in which points are uncorrelated.

    :param Poly poly: A polynomial object.
    :param numpy.ndarray correlation_matrix: The correlation matrix associated with the joint distribution.

    **References**
        1. Melchers, R. E., (1945) Structural Reliability Analysis and Predictions. John Wiley and Sons, second edition.

    """
    def __init__(self, poly, correlation_matrix, verbose=False):
        self.poly = poly
        D = self.poly.get_parameters()
        self.D = D
        self.R = correlation_matrix
        self.std = Parameter(order=5, distribution='normal',shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        inf_lim = -8.0
        sup_lim = - inf_lim
        p1 = Parameter(distribution = 'uniform', lower = inf_lim, upper = sup_lim, order = 31)
        myBasis = Basis('tensor-grid')
        self.Pols = Poly([p1, p1], myBasis, method='numerical-integration')
        Pols = self.Pols
        p = Pols.get_points()
        # w = Pols.get_weights()
        w = Pols.get_weights() * (sup_lim - inf_lim)**2
        p1 = p[:,0]
        p2 = p[:,1]
        R0 = np.eye((len(self.D)))
        for i in range(len(self.D)):
            for j in range(i+1, len(self.D), 1):
                if self.R[i,j] == 0:
                    R0[i,j] = 0.0
                else:
                    z1 = np.array(self.D[i].get_icdf(self.std.get_cdf(points=p1)))
                    z2 = np.array(self.D[j].get_icdf(self.std.get_cdf(points=p2)))

                    tp11 = (z1 - self.D[i].mean) / np.sqrt( self.D[i].variance )
                    tp22 = (z2 - self.D[j].mean)/ np.sqrt( self.D[j].variance )

                    coefficientsIntegral = np.flipud(tp11*tp22 * w)
                    def check_difference(rho_ij):
                        bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                        diff = np.dot(coefficientsIntegral, bivariateNormalPDF)
                        return diff - self.R[i,j]

                    if (self.D[i].name!='custom') or (self.D[j].name!='custom'):
                        rho = optimize.newton(check_difference, self.R[i,j], maxiter=50)
                    else:
                        # ???
                        res = optimize.least_squares(check_difference, self.R[i,j], bounds=(-0.999,0.999), ftol=1.e-03)
                        rho = res.x
                        print('A Custom Marginal is present')

                    R0[i,j] = rho
                    R0[j,i] = R0[i,j]
        self.R0 = R0.copy()

        self.A = np.linalg.cholesky(R0)
        if verbose is True:
            print('The Cholesky decomposition of fictive matrix R0 is:')
            print(self.A)
            print('The fictive matrix is:')
            print(R0)

        # Why generate quad points from normal, not from original marginals?
        list_of_parameters = []
        for i in range(0, len(self.D)):
            standard_parameter = Parameter(order=self.D[i].order, distribution='gaussian', shape_parameter_A = 0., shape_parameter_B = 1.)
            list_of_parameters.append(standard_parameter)

        # have option so that we don't need to obtain
        self.polystandard = deepcopy(self.poly)

        if hasattr(self.polystandard, '_quadrature_points'):
            self.standard_samples = self.polystandard._quadrature_points
            self.polystandard._set_parameters(list_of_parameters)
            self._points = self.get_correlated_from_uncorrelated(self.standard_samples)
    def get_points(self):
        """
        Returns the correlated samples based on the quadrature rules used in poly.

        :param Correlations self: An instance of the Correlations object.

        :return:
            **points**: A numpy.ndarray of sampled quadrature points with shape (number_of_samples, dimension).

        """
        return self._points
    def set_model(self, model=None, model_grads=None):
        """
        Computes the coefficients of the polynomial.

        :param Correlations self:
            An instance of the Correlations class.
        :param callable model:
            The function that needs to be approximated. In the absence of a callable function, the input can be the function evaluated at the quadrature points.
        :param callable model_grads:
            The gradient of the function that needs to be approximated. In the absence of a callable gradient function, the input can be a matrix of gradient evaluations at the quadrature points.
        """
        # Need to account for the nataf transform here?
        model_values = None
        model_grads_values = None
        if callable(model):
            model_values = evaluate_model(self._points, model)
        else:
            model_values = model
        if model_grads is not None:
            if callable(model_grads):
                model_grads_values = evaluate_model_gradients(self._points, model_grads)
            else:
                model_grads_values = model_grads
        self.polystandard.set_model(model_values, model_grads_values)
    def get_transformed_poly(self):
        """
        Returns the transformed polynomial.

        :param Correlations self:
            An instance of the Correlations class.

        :return:
            **poly**: An instance of the Poly class.
        """
        return self.polystandard
    def get_correlated_from_uncorrelated(self, X):
        """
        Method for mapping uncorrelated variables from standard normal space to a new physical space in which variables are correlated.

        :param Correlations self: An instance of the Correlations object.
        :param numpy.ndarray X: Samples of uncorrelated points from the marginals; of shape (N,M)

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the correlated samples.
        """
        # This seems to assume original marginal densities are gaussian?
        N = X.shape[0]
        d = len(self.D)
        # convert to gaussian here...
        # It should *not* be used to convert quad points... because they aren't
        # distributed as the marginal distribution! (Or should it?)
        # It happens to work with Gaussian quadrature points for Gaussian densities
        # because the icdf matches exactly and it becomes only a linear transform

        W = X @ self.A.T

        U = np.zeros((N, d))
        for i in range(d):
            U[:,i] = self.std.get_cdf(points=W[:,i])
        Z = np.zeros((N, d))
        for i in range(d):
            Z[:,i] = self.D[i].get_icdf(U[:,i])
        return Z
    def get_correlated_samples(self, N):
        """
        Method for generating correlated samples.

        :param int N: Number of correlated samples required.

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the correlated samples.
        """
        d = len(self.D)
        X_test = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=N)

        X_test = X_test @ self.A.T

        U_test = np.zeros(X_test.shape)
        for i in range(d):
            U_test[:, i] = self.std.get_cdf(X_test[:, i])

        Z_test = np.zeros(U_test.shape)
        for i in range(d):
            Z_test[:, i] = self.D[i].get_icdf(U_test[:, i])
        return Z_test

    def get_pdf(self, X):
        """
        Evaluate PDF at the sample points.
        :param numpy.ndarray X: Sample points (Number of points by dimensions)
        :return:
            **C**: A numpy.ndarray of shape (N,) with evaluations of the PDF.
        """

        poly = self.poly
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        d = X.shape[1]
        parameters = poly.get_parameters()
        U = np.zeros(X.shape)
        for i in range(d):
            U[:, i] = norm.ppf(parameters[i].get_cdf(X[:, i]))
        cop_num = multivariate_normal(mean=np.zeros(d), cov=self.R0).pdf(U)
        cop_den = np.prod(np.array([norm.pdf(U[:, i]) for i in range(d)]), axis=0)
        marginal_prod = np.prod(np.array([parameters[i].get_pdf(X[:, i]) for i in range(d)]), axis=0)
        return cop_num / cop_den * marginal_prod
