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
from equadratures.distributions.custom import Custom
import numpy as np

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
        `pareto <https://en.wikipedia.org/wiki/Pareto_distribution>`_. If no string is provided, a ``uniform`` distribution is assumed. If the user provides data, and would like to generate orthogonal
        polynomials (and quadrature rules) based on the data, they can set this option to be ``custom``.
    :param float shape_parameter_A:
        Most of the aforementioned distributions are characterized by two shape parameters. For instance, in the case of a ``gaussian`` (or ``truncated-gaussian``), this represents the mean. In the case of a beta distribution this represents the alpha value. For a ``uniform`` distribution this input is not required.
    :param float shape_parameter_B:
        This is the second shape parameter that characterizes the distribution selected. In the case of a ``gaussian`` or ``truncated-gaussian``, this is the variance.
    :param numpy.ndarray data:
        A data-set with shape (number_of_data_points, 2), where the first column comprises of parameter values, while the second column corresponds to the data observations. This input should only be used with the ``custom`` distribution.
    :param bool endpoints:
        If set to ``True``, then the quadrature points and weights will have end-points, based on Gauss-Lobatto quadrature rules.
    """
    def __init__(self, order, distribution, endpoints=False, shape_parameter_A=None, shape_parameter_B=None, lower=None, upper=None, data=None):
        self.name = distribution
        self.order = order
        self.shape_parameter_A = shape_parameter_A
        self.shape_parameter_B = shape_parameter_B
        self.lower = lower
        self.upper = upper
        self.endpoints = endpoints
        self.data = data
        self.__set_distribution()
        self.__set_bounds()
        self.__set_moments()
    def __set_distribution(self):
        """
        Private function that sets the distribution.
        :param Parameter self:
            An instance of the Parameter object.
        """
        choices = {'gaussian': Gaussian(self.shape_parameter_A, self.shape_parameter_B),
                   'normal': Gaussian(self.shape_parameter_A, self.shape_parameter_B),
                   'uniform' : Uniform(self.lower, self.upper),
                   'custom': Custom(self.data),
                   'beta': Beta(self.lower, self.upper, self.shape_parameter_A, self.shape_parameter_B),
                   'cauchy' : Cauchy(self.shape_parameter_A, self.shape_parameter_B),
                   'exponential': Exponential(self.shape_parameter_A),
                   'gamma': Gamma(self.shape_parameter_A, self.shape_parameter_B),
                   'weibull': Weibull(self.shape_parameter_A, self.shape_parameter_B),
                   'arcsine': Chebyshev(self.lower, self.upper),
                   'chebyshev': Chebyshev(self.lower, self.upper),
                   'rayleigh' : Rayleigh(self.shape_parameter_A),
                   'chisquared' : Chisquared(self.shape_parameter_A),
                   'truncated-gaussian': TruncatedGaussian(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
                   }
        distribution = choices.get(self.name.lower(), distribution_error)
        self.distribution = distribution
    def __set_moments(self):
        """
        Private function that sets the mean and the variance of the distribution.
        :param Parameter self:
            An instance of the Parameter object.
        """
        self.mean = self.distribution.mean
        self.variance = self.distribution.variance
    def __set_bounds(self):
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
            D,V = np.linalg.eig(self.get_jacobi_matrix(order))
            V = np.mat(V) # convert to matrix
            i = np.argsort(D) # get the sorted indices
            i = np.array(i) # convert to array
            V = V[:,i]
        return V
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

    def _get_local_quadrature(self, order=None):
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
        if self.endpoints is False:
            return get_local_quadrature(self, order, ab)
        elif self.endpoints is True:
            return get_local_quadrature_lobatto(self, order, ab)
        else:
            raise(ValueError, '_get_local_quadrature:: Error with Endpoints entry!')

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
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        local_points = np.sort(D) # sort by the eigenvalues
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        w = np.linspace(1,order+1,order) # create space for weights
        p = np.ones((int(order),1))
        for u in range(0, len(i) ):
            w[u] = ab[0,1] * (V[0,i[u]]**2) # replace weights with right value
            p[u,0] = local_points[u]
            if (p[u,0] < 1e-16) and (-1e-16 < p[u,0]):
                p[u,0] = np.abs(p[u,0])
    return p, w

def get_local_quadrature_lobatto(self, order=None, ab=None):
    # Check for extra input argument!
    if order is None:
        order = self.order - 2
    else:
        order = order - 2
    a = self.distribution.shape_parameter_A
    b = self.distribution.shape_parameter_B
    N = order
    # Get the recurrence coefficients & the jacobi matrix
    if ab is None:
        ab = self.get_recurrence_coefficients(order+2)
    else:
        ab = ab[0:order+2, :]
    ab[N+2, 0] = (a - b) / (2 * float(N+1) + a + b + 2)
    ab[N+2, 1] = 4 * (float(N+1) + a + 1) * (float(N+1) + b + 1) * (float(N+1) + a + b + 1) / ((2 * float(N+1) + a + b + 1) *
    (2 * float(N+1) + a + b + 2)**2)
    K = N + 2
    n0, __ = ab.shape
    if n0 < K:
        raise(ValueError, 'getlocalquadraturelobatto: Recurrence coefficients size misalignment!')
    J = np.zeros((K+1,K+1))
    for n in range(0, K+1):
        J[n,n] = ab[n, 0]
    for n in range(1, K+1):
        J[n, n-1] = np.sqrt(ab[n,1])
        J[n-1, n] = J[n, n-1]
    D, V = np.linalg.eig(J)
    V = np.mat(V) # convert to matrix
    local_points = np.sort(D) # sort by the eigenvalues
    i = np.argsort(D) # get the sorted indices
    i = np.array(i) # convert to array
    w = np.linspace(1,K+1,K+1) # create space for weights
    p = np.ones((int(K+1),1))
    for u in range(0, len(i) ):
        w[u] = ab[0,1] * (V[0,i[u]]**2) # replace weights with right value
        p[u,0] = local_points[u]
    return p, w

def distribution_error():
    raise(ValueError, 'Please select a valid distribution for your parameter; documentation can be found at www.effective-quadratures.org')
