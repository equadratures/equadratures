"Definition of a univariate parameter."
from equadratures.distributions.gaussian import Gaussian
from equadratures.distributions.uniform import Uniform
from equadratures.distributions.chebyshev import Chebyshev
from equadratures.distributions.beta import Beta
from equadratures.distributions.cauchy import Cauchy
from equadratures.distributions.exponential import Exponential
from equadratures.distributions.gamma import Gamma
from equadratures.distributions.weibull import Weibull
from equadratures.distributions.rayleigh import Rayleigh
from equadratures.distributions.chisquared import Chisquared
from equadratures.distributions.truncated_gaussian import TruncatedGaussian
from equadratures.distributions.pareto import Pareto
from equadratures.distributions.lognormal import Lognormal
from equadratures.distributions.studentst import Studentst
from equadratures.distributions.logistic import Logistic
from equadratures.distributions.gumbel import Gumbel
from equadratures.distributions.chi import Chi
from equadratures.distributions.custom import Custom
import numpy as np
import scipy as sc

class Parameter(object):
    """
    This class defines a univariate parameter. Below are details of its constructor.

    :param float lower: Lower bound for the parameter.
    :param float upper: Upper bound for the parameter.
    :param int order: Order of the parameter.
    :param str param_type:
        The type of distribution that characterizes the parameter. Options include `chebyshev (arcsine) <https://en.wikipedia.org/wiki/Arcsine_distribution>`_, `gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_,
        `truncated-gaussian <https://en.wikipedia.org/wiki/Truncated_normal_distribution>`_, `beta <https://en.wikipedia.org/wiki/Beta_distribution>`_,
        `cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_, `exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_,
        `uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_, `gamma <https://en.wikipedia.org/wiki/Gamma_distribution>`_,
        `weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_, `rayleigh  <https://en.wikipedia.org/wiki/Rayleigh_distribution>`_,
        `pareto <https://en.wikipedia.org/wiki/Pareto_distribution>`_, `lognormal <https://en.wikipedia.org/wiki/Log-normal_distribution>`_,
        `students-t <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_, `logistic <https://en.wikipedia.org/wiki/Log-normal_distribution>`_,
        `gumbel <https://en.wikipedia.org/wiki/Gumbel_distribution>`_, `chi <https://en.wikipedia.org/wiki/Chi_distribution>`_  and `chi-squared <https://en.wikipedia.org/wiki/Chi-squared_distribution>`_.
        If no string is provided, a ``uniform`` distribution is assumed. If the user provides data, and would like to generate orthogonal
        polynomials (and quadrature rules) based on the data, they can set this option to be ``custom`` (see [1, 2]).
    :param float shape_parameter_A:
        Most of the aforementioned distributions are characterized by two shape parameters. For instance, in the case of a ``gaussian`` (or ``truncated-gaussian``), this represents the mean. In the case of a beta distribution this represents the alpha value. For a ``uniform`` distribution this input is not required.
    :param float shape_parameter_B:
        This is the second shape parameter that characterizes the distribution selected. In the case of a ``gaussian`` or ``truncated-gaussian``, this is the variance.
    :param numpy.ndarray data:
        A data-set with shape (number_of_data_points, 2), where the first column comprises of parameter values, while the second column corresponds to the data observations. This input should only be used with the ``custom`` distribution.
    :param string endpoints:
        If set to ``both``, then the quadrature points and weights will have end-points, based on Gauss-Lobatto quadrature rules. If set to ``upper`` or ``lower`` a Gauss-Radau rule is used to compute one end-point at either the upper or lower bound.

    **Sample constructor initialisations**::

        import numpy as np
        from equadratures import *

        # uniform parameter.
        param = Parameter(distribution='uniform', lower=-2, upper=2., order=3)

        # beta parameter
        param = Parameter(distribution='beta', lower=-2., upper=15., order=4, shape_parameter_A=3.2, shape_parameter_B=1.7)

    **References**
        1. Xiu, D., Karniadakis, G. E., (2002) The Wiener-Askey Polynomial Chaos for Stochastic Differential Equations. SIAM Journal on Scientific Computing,  24(2), `Paper <https://epubs.siam.org/doi/abs/10.1137/S1064827501387826?journalCode=sjoce3>`__
        2. Gautschi, W., (1985) Orthogonal Polynomials-Constructive Theory and Applications. Journal of Computational and Applied Mathematics 12 (1985), pp. 61-76. `Paper <https://www.sciencedirect.com/science/article/pii/037704278590007X>`__
    """
    def __init__(self, order=1, distribution='Uniform', endpoints=None, shape_parameter_A=None, shape_parameter_B=None, variable='parameter', lower=None, upper=None, data=None):
        self.name = distribution
        self.variable = variable
        self.order = order
        self.shape_parameter_A = shape_parameter_A
        self.shape_parameter_B = shape_parameter_B
        self.lower = lower
        self.upper = upper
        self.endpoints = endpoints
        self.data = data
        self._set_distribution()
        self._set_bounds()
        self._set_moments()
        if self.endpoints is not None:
            if (self.distribution.bounds[0] == -np.inf) and (self.distribution.bounds[1] == np.inf) and (self.endpoints.lower() == 'both'):
                raise(ValueError, 'Parameter: The lower bound for your distribution is -infinity and the upper bound is infinity. Furthermore, you have selected the to have both endpoints. These options are incompatible!')
            if (self.distribution.bounds[0] == -np.inf) and (self.endpoints.lower() == 'lower'):
                raise(ValueError, 'Parameter: The lower bound for your distribution is -infinity and you have selected the lower bound option in the endpoints. These options are incompatible!')
            if (self.distribution.bounds[1] == np.inf) and (self.endpoints.lower() == 'upper'):
                raise(ValueError, 'Parameter: The upper bound for your distribution is infinity and you have selected the upper bound option in the endpoints. These options are incompatible!')
    def _set_distribution(self):
        """
        Private function that sets the distribution.

        :param Parameter self:
            An instance of the Parameter object.
        """
        if self.name.lower() == 'gaussian' or self.name.lower() == 'normal':
            self.distribution = Gaussian(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'uniform':
            self.distribution = Uniform(self.lower, self.upper)
        elif self.name.lower() == 'custom':
            self.distribution = Custom(self.data)
        elif self.name.lower() == 'beta':
            self.distribution = Beta(self.lower, self.upper, self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'truncated-gaussian':
            self.distribution = TruncatedGaussian(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.name.lower() == 'cauchy':
            self.distribution = Cauchy(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'exponential':
            self.distribution = Exponential(self.shape_parameter_A)
        elif self.name.lower() == 'gamma':
            self.distribution = Gamma(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'weibull':
            self.distribution = Weibull(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'arcsine' or self.name.lower() == 'chebyshev':
            self.distribution = Chebyshev(self.lower, self.upper)
        elif self.name.lower() == 'rayleigh':
            self.distribution = Rayleigh(self.shape_parameter_A)
        elif self.name.lower() == 'chi-squared':
            self.distribution = Chisquared(self.shape_parameter_A)
        elif self.name.lower() == 'chi':
            self.distribution = Chi(self.shape_parameter_A)
        elif self.name.lower() == 'pareto':
            self.distribution = Pareto(self.shape_parameter_A)
        elif self.name.lower() == 'gumbel':
            self.distribution = Gumbel(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'logistic':
            self.distribution = Logistic(self.shape_parameter_A, self.shape_parameter_B)
        elif self.name.lower() == 'students-t' or self.name.lower() == 't' or self.name.lower() == 'studentt':
            self.distribution = Studentst(self.shape_parameter_A)
        elif self.name.lower() == 'lognormal' or self.name.lower() == 'log-normal':
            self.distribution = Lognormal(self.shape_parameter_A)
        else:
            distribution_error()
    def _set_moments(self):
        """
        Private function that sets the mean and the variance of the distribution.

        :param Parameter self:
            An instance of the Parameter object.
        """
        self.mean = self.distribution.mean
        self.variance = self.distribution.variance
    def _set_bounds(self):
        """
        Private function that sets the bounds of the distribution.

        :param Parameter self:
            An instance of the Parameter object.
        """
        self.bounds = self.distribution.bounds
    def get_pdf(self, points=None):
        """
        Computes the probability density function associated with the Parameter.

        :param Parameter self:
            An instance of the Parameter object.
        :param numpy.ndarray points:
            Values of the parameter at which the PDF must be evaluated.
        """
        if points is None:
            x = self.distribution.x_range_for_pdf
            return x, self.distribution.get_pdf(x)
        else:
            return self.distribution.get_pdf(points)
    def get_cdf(self, points=None):
        """
        Computes the cumulative density function associated with the Parameter.

        :param Parameter self:
            An instance of the Parameter object.
        :param numpy.ndarray points:
            Values of the parameter at which the PDF must be evaluated.
        """
        if points is None:
            x = self.distribution.x_range_for_pdf
            return x, self.distribution.get_cdf(x)
        else:
            return self.distribution.get_cdf(points)
    def get_icdf(self, cdf_values):
        """
        Computes the inverse cumulative density function associated with the Parameter.

        :param Parameter self:
            An instance of the Parameter object.
        :param numpy.ndarray cdf_values:
            Values of the cumulative density function for which its inverse needs to be computed.
        """
        return self.distribution.get_icdf(cdf_values)
    def get_samples(self, number_of_samples_required):
        """
        Generates samples from the distribution associated with the Parameter.

        :param Parameter self:
            An instance of the Parameter object.
        :param int number_of_samples_required:
            Number of samples that are required.
        """
        return self.distribution.get_samples(number_of_samples_required)
    def get_description(self):
        """
        Provides a description of the Parameter.

        :param Parameter self:
            An instance of the Parameter object.
        """
        return self.distribution.get_description()
    def get_recurrence_coefficients(self, order=None):
        """
        Generates the recurrence coefficients.

        :param Parameter self:
            An instance of the Parameter object.
        :param int order:
            Order of the recurrence coefficients.
        """
        return self.distribution.get_recurrence_coefficients(order)
    def get_jacobi_eigenvectors(self, order=None):
        """
        Computes the eigenvectors of the Jacobi matrix.

        :param Parameter self:
            An instance of the Parameter object.
        :param int order:
            Order of the recurrence coefficients.
        """
        if order is None:
            order = self.order + 1
            JacobiMat = self.get_jacobi_matrix(order)
            if order == 1:
                V = [1.0]
        else:
            #D,V = np.linalg.eig(self.get_jacobi_matrix(order))
            D, V = sc.linalg.eigh(JacobiMat)
            idx = D.argsort()[::-1]
            eigs = D[idx]
            eigVecs = V[:, idx]
            #V = np.mat(V) # convert to matrix
            #i = np.argsort(D) # get the sorted indices
            #i = np.array(i) # convert to array
            #V = V[:,i]
        return eigVecs
    def get_jacobi_matrix(self, order=None, ab=None):
        """
        Computes the Jacobi matrix---a tridiagonal matrix of the recurrence coefficients.

        :param Parameter self:
            An instance of the Parameter object.
        :param int order:
            Order of the recurrence coefficients.
        """
        if order is None and ab is None:
            ab = self.get_recurrence_coefficients()
            order = self.order + 1
        elif ab is None:
            ab = self.get_recurrence_coefficients(order)
        else:
            ab = ab[0:order, :]

        order = int(order)
        # The case of order 1~
        if int(order) == 1:
            JacobiMatrix = ab[0, 0]
        # For everything else~
        else:
            JacobiMatrix = np.zeros((int(order), int(order))) # allocate space
            JacobiMatrix[0,0] = ab[0,0]
            JacobiMatrix[0,1] = np.sqrt(ab[1,1])
            k = order - 1
            for u in range(1, int(k)):
                JacobiMatrix[u,u] = ab[u,0]
                JacobiMatrix[u,u-1] = np.sqrt(ab[u,1])
                JacobiMatrix[u,u+1] = np.sqrt(ab[u+1,1])

            JacobiMatrix[order-1, order-1] = ab[order-1,0]
            JacobiMatrix[order-1, order-2] = np.sqrt(ab[order-1,1])
        return JacobiMatrix
    def _get_orthogonal_polynomial(self, points, order=None):
        """
        Private function that evaluates the univariate orthogonal polynomial at quadrature points.

        :param Parameter self:
            An instance of the Parameter object.
        :param numpy.ndarray points:
            Points at which the orthogonal polynomial must be evaluated.
        :param int order:
            Order up to which the orthogonal polynomial must be obtained.
        """
        if order is None:
            order = self.order + 1
        else:
            order = order + 1
        gridPoints = np.asarray(points).copy()
        ab = self.get_recurrence_coefficients(order)
        if (any(gridPoints) < self.bounds[0]) or (any(gridPoints) > self.bounds[1]):
            for r in range(0, len(gridPoints)):
                gridPoints[r] = (gridPoints[r] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        orthopoly = np.zeros((order, len(gridPoints)))  # create a matrix full of zeros
        derivative_orthopoly = np.zeros((order, len(gridPoints)))
        dderivative_orthopoly = np.zeros((order, len(gridPoints)))

        # Convert the grid points to a numpy array -- simplfy life!
        gridPointsII = np.zeros((len(gridPoints), 1))
        for u in range(0, len(gridPoints)):
            gridPointsII[u, 0] = gridPoints[u]
        orthopoly[0, :] = 1.0

        # Cases
        if order == 1:  # CHANGED 2/2/18
            return orthopoly, derivative_orthopoly, dderivative_orthopoly
        orthopoly[1, :] = ((gridPointsII[:, 0] - ab[0, 0]) * orthopoly[0, :]) * (1.0) / (1.0 * np.sqrt(ab[1, 1]))
        derivative_orthopoly[1, :] = orthopoly[0, :] / (np.sqrt(ab[1, 1]))
        if order == 2:  # CHANGED 2/2/18
            return orthopoly, derivative_orthopoly, dderivative_orthopoly

        if order >= 3:  # CHANGED 2/2/18
            for u in range(2, order):  # CHANGED 2/2/18
                # Three-term recurrence rule in action!
                orthopoly[u, :] = (((gridPointsII[:, 0] - ab[u - 1, 0]) * orthopoly[u - 1, :]) - np.sqrt(
                    ab[u - 1, 1]) * orthopoly[u - 2, :]) / (1.0 * np.sqrt(ab[u, 1]))
            for u in range(2, order):  # CHANGED 2/2/18
                # Four-term recurrence formula for derivatives of orthogonal polynomials!
                derivative_orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
            for u in range(2,order):
                # Four-term recurrence formula for second derivatives of orthogonal polynomials!
                dderivative_orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0]) * dderivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * dderivative_orthopoly[u-2,:] ) +  2.0 * derivative_orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))

        return orthopoly, derivative_orthopoly, dderivative_orthopoly
    def _get_local_quadrature(self, order=None, ab=None):
        """
        Returns the 1D quadrature points and weights for the parameter. WARNING: Should not be called under normal circumstances.

        :param Parameter self:
            An instance of the Parameter class
        :param int N:
            Number of quadrature points and weights required. If order is not specified, then by default the method will return the number of points defined in the parameter itself.
        :return:
            A N-by-1 matrix that contains the quadrature points
        :return:
            A 1-by-N matrix that contains the quadrature weights
        """
        if self.endpoints is None:
            return get_local_quadrature(self, order, ab)
        elif self.endpoints.lower() == 'lower' or self.endpoints.lower() == 'upper':
            return get_local_quadrature_radau(self, order, ab)
        elif self.endpoints.lower() == 'both':
            return get_local_quadrature_lobatto(self, order, ab)
        else:
            raise(ValueError, 'Error in endpoints specification.')
def get_local_quadrature(self, order=None, ab=None):
    # Check for extra input argument!
    if order is None:
        order = self.order + 1
    else:
        order = order + 1

    if ab is None:
        # Get the recurrence coefficients & the jacobi matrix
        JacobiMat = self.get_jacobi_matrix(order)
        ab = self.get_recurrence_coefficients(order+1)
    else:
        ab = ab[0:order+1,:]
        JacobiMat = self.get_jacobi_matrix(order, ab)
    # If statement to handle the case where order = 1
    if order == 1:
        # Check to see whether upper and lower bound are defined:
        if not self.lower or not self.upper:
            p = np.asarray(self.distribution.mean).reshape((1,1))
        else:
            p = np.asarray((self.upper - self.lower)/(2.0) + self.lower).reshape((1,1))
        w = [1.0]
    else:
        # Compute eigenvalues & eigenvectors of Jacobi matrix
        #D,V = np.linalg.eig(JacobiMat)
        D, V = sc.linalg.eigh(JacobiMat)
        #V = np.mat(V) # convert to matrix
        #local_points = np.sort(D) # sort by the eigenvalues
        #i = np.argsort(D) # get the sorted indices
        #i = np.array(i) # convert to array
        idx = D.argsort()[::-1]
        eigs = D[idx]
        eigVecs = V[:, idx]

        w = np.linspace(1,order+1,order) # create space for weights
        p = np.ones((int(order),1))
        for u in range(0, len(idx) ):
            w[u] = float(ab[0,1]) * (eigVecs[0,idx[u]]**2) # replace weights with right value
            p[u,0] = eigs[u]
            #if (p[u,0] < 1e-16) and (-1e-16 < p[u,0]):
            #    p[u,0] = np.abs(p[u,0])
    return p, w
def get_local_quadrature_radau(self, order=None, ab=None):
    if self.endpoints.lower() == 'lower':
        end0 = self.lower
    elif self.endpoints.lower() == 'upper':
        end0 = self.upper
    if order is None:
        order = self.order - 1
    else:
        order = order - 1
    N = order
    if ab is None:
        ab = self.get_recurrence_coefficients(order+1)
    else:
        ab = ab[0:order+1, :]
    p0 = 0.
    p1 = 1.
    for i in range(0, N+1):
        pm1 = p0
        p0 = p1
        p1 = (end0 - ab[i, 0]) * p0 - ab[i, 1]*pm1
    ab[N+1, 0] = end0 - ab[N+1, 1] * p0/p1
    return get_local_quadrature(self, order=order+1, ab=ab)
def get_local_quadrature_lobatto(self, order=None, ab=None):
    if order is None:
        order = self.order - 2
    else:
        order = order - 2
    N = order
    endl = self.lower
    endr = self.upper
    if ab is None:
        ab = self.get_recurrence_coefficients(order+2)
    else:
        ab = ab[0:order+2, :]
    p0l = 0.
    p0r = 0.
    p1l = 1.
    p1r = 1.
    for i in range(0, N+2):
        pm1l = p0l
        p0l = p1l
        pm1r = p0r
        p0r = p1r
        p1l = (endl - ab[i, 0]) * p0l - ab[i, 1] * pm1l
        p1r = (endr - ab[i, 0]) * p0r - ab[i, 1] * pm1r
    det = p1l * p0r - p1r * p0l
    ab[N+2, 0] = (endl*p1l*p0r-endr*p1r*p0l)/det
    ab[N+2, 1] = (endr - endl) * p1l * p1r/det
    return get_local_quadrature(self, order=order+2, ab=ab)
def distribution_error():
    raise(ValueError, 'Please select a valid distribution for your parameter; documentation can be found at www.effective-quadratures.org')
