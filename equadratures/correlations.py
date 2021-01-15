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
    The class defines methods for polynomial approximations with correlated inputs, including the Nataf transform and Gram-Schmidt process.

    :param numpy.ndarray correlation_matrix: The correlation matrix associated with the joint distribution.
    :param Poly poly: Polynomial defined with parameters with marginal distributions in uncorrelated space.
    :param list parameters: List of parameters with marginal distributions.
    :param str method: `nataf-transform` or `gram-schmidt`.
    :param bool verbose: Display Cholesky decomposition of the fictive matrix.

    **Example usage**::
        def func(x):
            s1 = s[0]
            s2 = s[1]
            return s1**2 - s2**3 - 2. * s1 - 0.5 * s2

        parameters = [Parameter(distribution='beta', shape_parameter_A=5.0, shape_parameter_B=2.0, lower=0.0, upper=1.0, order=15) for _ in range(2)]
        basis = Basis('total-order')
        uncorr_poly = Poly(parameters, basis, method='least-squares',
                            sampling_args={'mesh': 'monte-carlo',
                                            'subsampling-algorithm': 'lu'})

        corr_mat = np.array([[1.0, -0.855],
                            [-0.855, 1.0]])

        corr_gs = Correlations(corr_mat, poly=uncorr_poly, method='gram-schmidt')
        corr_gs.set_model(func)
        corrected_poly_gs = corr_gs.get_transformed_poly()
        print(corrected_poly_gs.get_mean_and_variance())

    **References**
        1. Melchers, R. E., (1945) Structural Reliability Analysis and Predictions. John Wiley and Sons, second edition.
        2. Jakeman, J. D. et al., (2019) Polynomial chaos expansions for dependent random variables.
    """
    def __init__(self, correlation_matrix, poly=None, parameters=None, method=None, verbose=False):
        if (poly is None) and (method is not None):
            raise ValueError('Need to specify poly for probability transform.')
        if poly is not None:
            self.poly = poly
            D = self.poly.get_parameters()
        elif parameters is not None:
            D = parameters
        else:
            raise ValueError('Need to specify either poly or parameters.')
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

                    # if (self.D[i].name!='custom') or (self.D[j].name!='custom'):
                    rho = optimize.newton(check_difference, self.R[i,j], maxiter=50)
                    # else:
                    #     # ???
                    #     res = optimize.least_squares(check_difference, self.R[i,j], bounds=(-0.999,0.999), ftol=1.e-03)
                    #     rho = res.x
                    #     print('A Custom Marginal is present')

                    R0[i,j] = rho
                    R0[j,i] = R0[i,j]
        self.R0 = R0.copy()

        self.A = np.linalg.cholesky(R0)
        if verbose:
            print('The Cholesky decomposition of fictive matrix R0 is:')
            print(self.A)
            print('The fictive matrix is:')
            print(R0)

        if method is None:
            pass
        elif method.lower() == 'nataf-transform':
            list_of_parameters = []
            for i in range(0, len(self.D)):
                standard_parameter = Parameter(order=self.D[i].order, distribution='gaussian', shape_parameter_A = 0., shape_parameter_B = 1.)
                list_of_parameters.append(standard_parameter)

            # have option so that we don't need to obtain
            self.corrected_poly = deepcopy(self.poly)

            if hasattr(self.corrected_poly, '_quadrature_points'):
                self.corrected_poly._set_parameters(list_of_parameters)
                self.standard_samples = self.corrected_poly._quadrature_points
                self._points = self.get_correlated_samples(X=self.standard_samples)
                # self.corrected_poly._quadrature_points = self._points.copy()
        elif method.lower() == 'gram-schmidt':
            basis_card = poly.basis.cardinality
            oversampling = 10

            N_Psi = oversampling * basis_card
            S_samples = self.get_correlated_samples(N=N_Psi)
            w_weights = 1.0 / N_Psi * np.ones(N_Psi)
            Psi = poly.get_poly(S_samples).T
            WPsi = np.diag(np.sqrt(w_weights)) @ Psi
            self.WPsi = WPsi

            R_Psi = np.linalg.qr(WPsi)[1]

            self.R_Psi = R_Psi
            self.R_Psi[0, :] *= np.sign(self.R_Psi[0, 0])
            self.corrected_poly = deepcopy(poly)
            self.corrected_poly.inv_R_Psi = np.linalg.inv(self.R_Psi)
            self.corrected_poly.corr = self
            self.corrected_poly._set_points_and_weights()

            P = self.corrected_poly.get_poly(self.corrected_poly._quadrature_points)
            W = np.mat(np.diag(np.sqrt(self.corrected_poly._quadrature_weights)))
            A = W * P.T
            self.corrected_poly.A = A
            self.corrected_poly.P = P

            if hasattr(self.corrected_poly, '_quadrature_points'):
                # TODO: Correlated quadrature points?
                self._points = self.corrected_poly._quadrature_points
        else:
            raise ValueError('Invalid method for correlations.')
    def get_points(self):
        """
        Returns the quadrature points accounting for correlations. For Nataf transform, returns points in the correlated standard normal space.
        For Gram-Schmidt, returns points according to the GS polynomial basis.

        :param Correlations self: An instance of the Correlations object.

        :return:
            **points**: A numpy.ndarray of quadrature points with shape (number_of_samples, dimension).

        """
        return self._points
    def set_model(self, model=None, model_grads=None):
        """
        Computes the coefficients of transformed polynomial (equivalent to calling `self.get_transformed_poly().set_model(...)`)

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
        self.corrected_poly.set_model(model_values, model_grads_values)
    def get_transformed_poly(self):
        """
        Returns the transformed polynomial.

        :param Correlations self:
            An instance of the Correlations class.

        :return:
            **poly**: An instance of the Poly class.
        """
        return self.corrected_poly
    def get_correlated_samples(self, N=None, X=None):
        """
        Method for generating correlated samples.

        :param int N: Number of correlated samples required.
        :param ndarray X: (Optional) Points in the uncorrelated space to map to the correlated space.

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the correlated samples.
        """
        d = len(self.D)

        if X is None:
            if N is None:
                raise ValueError('Need to specify number of points to generate.')
            X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=N)

        X_test = X @ self.A.T

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

        parameters = self.D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        d = X.shape[1]

        U = np.zeros(X.shape)
        for i in range(d):
            U[:, i] = norm.ppf(parameters[i].get_cdf(X[:, i]))
        cop_num = multivariate_normal(mean=np.zeros(d), cov=self.R0).pdf(U)
        cop_den = np.prod(np.array([norm.pdf(U[:, i]) for i in range(d)]), axis=0)
        marginal_prod = np.prod(np.array([parameters[i].get_pdf(X[:, i]) for i in range(d)]), axis=0)
        return cop_num / cop_den * marginal_prod
