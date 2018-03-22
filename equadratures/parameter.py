"""Core class for setting the properties of a univariate parameter."""
import numpy as np
from scipy.special import gamma
import distributions as analytical
class Parameter(object):
    """
    This class defines a univariate parameter. Below are details of its constructor.

    :param double lower:
        Lower bound for the parameter.
    :param double upper:
        Upper bound for the parameter.
    :param integer order:
        Order of the parameter.
    :param string param_type:
        The type of distribution that characterizes the parameter. Options include: `Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, `Truncated-Gaussian <https://en.wikipedia.org/wiki/Truncated_normal_distribution>`_, `Beta <https://en.wikipedia.org/wiki/Beta_distribution>`_, `Cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_, `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_, `Uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_, `Gamma <https://en.wikipedia.org/wiki/Gamma_distribution>`_, `Weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_. If no string is provided, a `Uniform` distribution is assumed. If the user provides data, and would like to generate orthogonal polynomials (and quadrature rules) based on the data, they can set this option to be Custom.
    :param double shape_parameter_A:
        Most of the aforementioned distributions are characterized by two shape parameters. For instance, in the case of a `Gaussian` (or `TruncatedGaussian`), this represents the mean. In the case of a Beta distribution this represents the alpha value. For a uniform distribution this input is not required.
    :param double shape_parameter_B:
        This is the second shape parameter that characterizes the distribution selected. In the case of a `Gaussian` or `TruncatedGaussian`, this is the variance.
    :param data:
        A numpy array with data values (x-y column format). Note this option is only invoked if the user uses the Custom param_type.
    """

    # constructor
    def __init__(self, order, lower=None, upper=None, param_type=None, shape_parameter_A=None, shape_parameter_B=None, data=None):
        self.order = order

        if param_type is None:
            self.param_type = 'Uniform'
        else:
            self.param_type = param_type

        if lower is None and data is None:
            if self.param_type is "Exponential":
                self.lower = 0.0
            else:
                self.lower = -1.0
        else:
            self.lower = lower

        if upper is None and data is None:
            self.upper = 1.0
        else:
            self.upper = upper

        if shape_parameter_A is None:
            self.shape_parameter_A = 0
        else:
            self.shape_parameter_A = shape_parameter_A

        if shape_parameter_B is None:
            self.shape_parameter_B = 0
        else:
            self.shape_parameter_B = shape_parameter_B

        if self.param_type == 'TruncatedGaussian' :
            if upper is None or lower is None:
                raise(ValueError, 'parameter __init__: upper and lower bounds are required for a TruncatedGaussian distribution!')

        if self.lower >= self.upper  and data is None:
            raise(ValueError, 'parameter __init__: upper bounds must be greater than lower bounds!')

        if data is not None:
            self.data = data
            if self.param_type != 'Custom':
                raise(ValueError, 'parameter __init__: if data is provided then the custom distribution must be selected!')
        self.bounds = None

    # Routine for computing the mean of the distributions
    def computeMean(self):
        """
        Returns the mean of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :return:
            Mean of the parameter.

        """
        if self.param_type == "Gaussian":
            mu = self.shape_parameter_A
        elif self.param_type == "TruncatedGaussian":
            mu = self.shape_parameter_A
        elif self.param_type == "Exponential":
            mu = 1.0/self.shape_parameter_A
        elif self.param_type == "Cauchy":
            mu = self.shape_parameter_A # technically the mean is undefined!
        elif self.param_type == "Weibull":
            mu = self.shape_parameter_A * gamma(1.0 + 1.0/self.shape_parameter_B)
        elif self.param_type == "Gamma":
            mu = self.shape_parameter_A * self.shape_parameter_B
        elif self.param_type == 'Custom':
            mu = np.mean(self.getSamples)
        return mu


    def getPDF(self, N):
        """
        Returns the probability density function of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer N:
            Number of points along the x-axis.
        :return:
            A 1-by-N matrix that contains the values of the x-axis along the support of the parameter.

        """
        if self.param_type is "Gaussian":
            x, y = analytical.PDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.PDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            x, y = analytical.PDF_GammaDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            x, y = analytical.PDF_WeibullDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            x, y = analytical.PDF_CauchyDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            x, y = analytical.PDF_UniformDistribution(N, self.lower, self.upper)
        elif self.param_type is "TruncatedGaussian":
            x, y = analytical.PDF_TruncatedGaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            x, y = analytical.PDF_ExponentialDistribution(N, self.shape_parameter_A)
        elif self.param_type is "Custom":
            x, y = analytical.PDF_CustomDistribution(N, self.data)
        else:
            raise(ValueError, 'parameter getPDF(): invalid parameter type!')
        return x, y

    def getSamples(self, m=None, graph=None):
        """
        Returns samples of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is None:
            number_of_random_samples = 500
        else:
            number_of_random_samples = m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.getiCDF(uniform_samples)
        return yy

    def getCDF(self, N):
        """
        Returns the cumulative density function of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer N:
            Number of points along the x-axis.
        :return:
            A 1-by-N matrix that contains the values of the x-axis along the support of the parameter.
        :return:
            A 1-by-N matrix that contains the values of the PDF of the parameter.

        """
        if self.param_type is "Gaussian":
            x, y = analytical.CDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.CDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            x, y = analytical.CDF_GammaDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            x, y = analytical.CDF_WeibullDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            x, y = analytical.CDF_CauchyDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            x, y = analytical.CDF_UniformDistribution(N, self.lower, self.upper)
        elif self.param_type is "TruncatedGaussian":
            x, y = analytical.CDF_TruncatedGaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            x, y = analytical.CDF_ExponentialDistribution(N, self.shape_parameter_A)
        else:
            raise(ValueError, 'parameter getCDF(): invalid parameter type!')
        return x, y

    def getiCDF(self, x):
        """
        Returns values of the inverse CDF.

        :param Parameter self:
            An instance of the Parameter class.
        :param numpy array:
            A 1-by-N array of doubles where each entry is between [0,1].
        :return:
            A 1-by-N array where each entry is the inverse CDF of input x.

        """
        if self.param_type is "Gaussian":
            y = analytical.iCDF_Gaussian(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            y = analytical.iCDF_BetaDistribution(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            y = analytical.iCDF_GammaDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            y = analytical.iCDF_WeibullDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            y = analytical.iCDF_CauchyDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            y = (x - .5) * (self.upper - self.lower) + (self.upper + self.lower)/2.0
        elif self.param_type is "TruncatedGaussian":
            y = analytical.iCDF_TruncatedGaussianDistribution(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            y = analytical.iCDF_ExponentialDistribution(x, self.shape_parameter_A)
        elif self.param_type is "Custom":
            y = analytical.iCDF_CustomDistribution(x, self.data)
        else:
            raise(ValueError, 'parameter getiCDF(): invalid parameter type!')
        return y

    def getRecurrenceCoefficients(self, order=None):
        """
        Returns the recurrence coefficients of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            The number of recurrence coefficients required. By default this is the same as the number of points used when the parameter constructor is initiated.
        :return:
            An order-by-2 matrix that containts the recurrence coefficients.

        """

        return recurrence_coefficients(self, order)

    def getJacobiMatrix(self, order=None):
        """
        Returns the tridiagonal Jacobi matrix.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            The number of rows and columns of the JacobiMatrix that is required. By default, this value is set to be the same as the number of points used when the parameter constructor is initiated.
        :return:
            An order-by-order sized Jacobi tridiagonal matrix.

        """
        return jacobiMatrix(self, order)

    def getJacobiEigenvectors(self, order=None):
        """
        Returns the eigenvectors of the tridiagonal Jacobi matrix. These are used for computing quadrature rules for numerical integration.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            Number of eigenvectors required. This function makes the call getJacobiMatrix(order) and then computes the corresponding eigenvectors.
        :return:
            A order-by-order matrix that contains the eigenvectors of the Jacobi matrix.
        """
        return jacobiEigenvectors(self, order)

    def _getOrthoPoly(self, points, order=None):
        """
        Returns orthogonal polynomials & its derivatives, evaluated at a set of points. WARNING: Should not be called under normal circumstances, without normalization of points!

        :param Parameter self:
            An instance of the Parameter class.
        :param ndarray points:
            Points at which the orthogonal polynomial (and its derivatives) should be evaluated at.
        :param int order:
            This value of order overwrites the order defined for the constructor.
        :return:
            An order-by-k matrix where order defines the number of orthogonal polynomials that will be evaluated and k defines the points at which these points should be evaluated at.
        :return:
            An order-by-k matrix where order defines the number of derivative of the orthogonal polynomials that will be evaluated and k defines the points at which these points should be evaluated at.
        """
        return orthoPolynomial_and_derivative(self, points, order)

    def induced_jacobi_distribution(self, x, order=None):
        """
        Evaluates the induced distribution.

        :param Parameter self:
            An instance of the Parameter class
        :param array x:
            ????
        :param int order:
            Order of the distribution. Note that this value will override the order associated with the Parameter instance.
        :return:
            The median estimate (double)
        """
        if order is None:
            order = self.order
        if self.param_type is 'Beta':
            alpha = self.shape_parameter_B - 1
            beta = self.shape_parameter_A - 1
        elif self.param_type is 'Uniform':
            alpha = 0.0
            beta = 0.0
        else:
            raise(ValueError: 'Parameter: median_approximation_jacobi:: Unrecognized parameter type!')

        # A few quick checks before we proceed!
        assert( (alpha > -1)) and (beta > -1) ):
        assert(all(np.abs(x)) <= 1)
        if len(x) == 0:
            F = []

        if M is None:
            M = 10
        A = np.floor(np.abs(alpha))
        Aa = alpha - A
        F = np.zeros((len(x)))
        centroid = self.median_approximation_jacobi()
        xreflect = np.ones((len(x)), dtype=bool)

        for i in range(0, len(xreflect)):
            if x[i] > centroid:
                xreflect[i] = 1
            else:
                xreflect[i] = 0

        F[xreflect] = 1 - self.induced_jacobi_distribution(-x[xreflect])
        ab = self.getRecurrenceCoefficients(self.order + 1)
        a = ab[:,0]
        b = ab[:,1]
        b[0,0] = 1.0 # Normalize to make it a probability measure

        if self.order > 0:
            xn, wn = self._getLocalQuadrature

        # Scaling factor for the inverse nth root of the leading coefficient square of pn
        kn_factor = np.exp(-1.0 / self.order * np.sum(np.log(b))  )

        for xq in range(0, len(x)):
            if x[xq] == -1:
                F[xq] = 0

            print xreflect[xq]

            # Recurrence coefficients for quadrature rule!
            neworder = 2 * self.order + A + M + 1
            ab = self.getRecurrenceCoefficients(neworder)
            ab[0,1] = 1 # To ensure that it is a probability measure!

            if self.order > 0:
                un = ( 2.0 / (x[xq] + 1.0) * (xn + 1.0) ) - 1.0

            logfactor = 0.0 # Keep this so that beta(1) always equals what it did before?

            # Sucessive quadratic measure modifications!
            for j in range(0, self.orders):
                ab = self._quadraticModification(un[j])
                logfactor = logfactor + np.log( b[0] *  ((x[xq] + 1.0)/2.0)**2 * kn_factor )
                b[1] = 1.0

            # Linear modification
            

    def median_approximation_jacobi(self, order=None):
        """
        Returns an estimate for the median of the order-n Jacobi induced distribution.

        :param Parameter self:
            An instance of the Parameter class
        :param int order:
            Order of the distribution. Note that this value will override the order associated with the Parameter instance.
        :return:
            The median estimate (double)
        """
        if order is None:
            order = self.order
        if self.param_type is 'Beta':
            alpha = self.shape_parameter_B - 1 # bug fix @ 9/6/2016
            beta = self.shape_parameter_A - 1
        elif self.param_type is 'Uniform':
            alpha = 0.0
            beta = 0.0
        else:
            raise(ValueError: 'Parameter: median_approximation_jacobi:: Unrecognized parameter type!')
        if n > 0 :
            x0 = (beta**2 - alpha**2) / (2 * order + alpha + beta)**2
        else:
            x0 = 2.0/(1.0 + (alpha + 1.0)/(beta + 1.0))  - 1.0
        return x0

    def _getLocalQuadrature(self, order=None, scale=None):
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
        return getlocalquadrature(self, order, scale)

    def _quadraticModification(self, x0):
        """
        Performs a quadratic modification of the orthogonal polynomial recurrence coefficients. It transforms the coefficients
        such that the new coefficients are associated with a polynomial family that is orthonormal under the weight (x - x0)**2

        :param Parameter self:
            An instance of the Parameter class
        :param double:
            The shift in the weights
        :return:
            A N-by-2 matrix that contains the modified recurrence coefficients.
        """
        N = self.order
        if N < 2:
            raise(ValueError, 'N should be greater than 2!')
        ab = np.zeros((N-2, N-2))
        alphabeta = self.getRecurrenceCoefficients()
        alpha = alphabeta[:,0]
        beta = alphabeta[:,1]
        C = np.reshape(  christoffelNormalizedOrthogonalPolynomials(alpha, beta, x0, N-1)  , [N, 1] )

        # q is the length N -- new coefficients have length N-2
        acorrect = np.zeros((N-2, 1))
        bcorrect = np.zeros((N-2, 1))
        temp = np.zeros((N-1))
        temp_2 = np.zeros((N-1))

        for i in range(2, N):
            j = i - 2
            temp[i-2] = np.sqrt(beta[i]) * C[i] * C[j] * 1.0/(np.sqrt(1 + C[j]**2 ))
        temp[0] = np.sqrt(beta[1]) * C[1] # A special case -- why?

        for i in range(0, N-2):
            acorrect[i] = temp[i + 1] - temp[i]

        for i in range(0, N-1):
            temp_2[i] = 1.0 + C[i]**2

        for i in range(0, N-2):
            bcorrect[i] = temp[i+1] / temp[i]
            if i == 0:
                bcorrect[0] = 1.0 + C[1]**2
            ab[i,0] = alpha[i+2] + acorrect[i,0]
            ab[i,1] = beta[i+2] * bcorrect[i]
        return ab


#-----------------------------------------------------------------------------------
#
#                               PRIVATE FUNCTIONS BELOW
#
#-----------------------------------------------------------------------------------
def christoffelNormalizedOrthogonalPolynomials(a, b, x, N):
    # Evaluates the Christoffel normalized orthogonal getPolynomialCoefficients
    nx = len(x)
    assert N>= 0
    assert N <= len(a)
    assert N <= len(b)

    C = np.zeros((nx, N+1))
    xf = x

    # Initialize the polynomials!
    C[:,0] = 1.0/ sqrt(b[0])
    if N > 0:
        for k in range(0, len(x)):
            C[k,1] = 1.0 / np.sqrt(b[1]) * (xf[k] - a[0])

    if N > 1:
        for k in range(0, len(x)):
            C[k,2] = 1.0 / np.sqrt(1.0 + C[k,1]**2)  * (  (xf[k] - a[1]) * C[k,1] - np.sqrt(b[1]) )
            C[k,2] = C[k,2] / np.sqrt(b[2])

    if N > 2:
        for n in range(2, N):
            for k in range(0, len(x)):
                C[k,n+1] = 1.0/np.sqrt(1.0 + C[k,n]**2) * (  (xf[k] - a[n]) * C[k,n] - np.sqrt(b[n]) ) * C[k,n-1] / np.sqrt(1.0 + C[k, n-1]**2)
                C[k,n+1] = C[k,n+1] / np.sqrt(b[n+1])
    return C


# Call different methods depending on the choice of the polynomial parameter
def recurrence_coefficients(self, order=None):

    # Preliminaries.
    N = 8000 # no. of points for analytical distributions.
    if order  is None:
        order = self.order

    # 1. Beta distribution
    if self.param_type is "Beta":
        param_A = self.shape_parameter_B - 1 # bug fix @ 9/6/2016
        param_B = self.shape_parameter_A - 1
        if(param_B <= 0):
            raise(ValueError, 'ERROR: parameter_A (beta shape parameter A) must be greater than 1!')
        if(param_A <= 0):
            raise(ValueError, 'ERROR: parameter_B (beta shape parameter B) must be greater than 1!')
        ab =  jacobi_recurrence_coefficients_01(param_A, param_B , order)
        #alpha = self.shape_parameter_A
        #beta = self.shape_parameter_B
        #lower = self.lower
        #upper = self.upper
        #x, w = analytical.PDF_BetaDistribution(N, alpha, beta, lower, upper)
        #ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0,1]

    # 2. Uniform distribution
    elif self.param_type is "Uniform":
        self.shape_parameter_A = 0.0
        self.shape_parameter_B = 0.0
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, order)
        self.bounds = [-1, 1]

    elif self.param_type is "Custom":
        x, w = analytical.PDF_CustomDistribution(N, self.data)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [np.min(x), np.max(x)]
        self.upper = np.max(x)
        self.lower = np.min(x)

    # 3. Analytical Gaussian defined on [-inf, inf]
    elif self.param_type is "Gaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        x, w  = analytical.PDF_GaussianDistribution(N, mu, sigma)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [-np.inf, np.inf]

    # 4. Analytical Exponential defined on [0, inf]
    elif self.param_type is "Exponential":
        lambda_value = self.shape_parameter_A
        x, w  = analytical.PDF_ExponentialDistribution(N, lambda_value)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 5. Analytical Cauchy defined on [-inf, inf]
    elif self.param_type is "Cauchy":
        x0 = self.shape_parameter_A
        gammavalue = self.shape_parameter_B
        x, w  = analytical.PDF_CauchyDistribution(N, x0, gammavalue)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [-np.inf, np.inf]

    # 5. Analytical Gamma defined on [0, inf]
    elif self.param_type is "Gamma":
        k = self.shape_parameter_A
        theta = self.shape_parameter_B
        x, w  = analytical.PDF_GammaDistribution(N, k, theta)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 6. Analytical Weibull defined on [0, inf]
    elif self.param_type is "Weibull":
        lambda_value= self.shape_parameter_A
        k = self.shape_parameter_B
        x, w  = analytical.PDF_WeibullDistribution(N, lambda_value, k)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 7. Analytical Truncated Gaussian defined on [a,b]
    elif self.param_type is "TruncatedGaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        a = self.lower
        b = self.upper
        x, w  = analytical.PDF_TruncatedGaussianDistribution(N, mu, sigma, a, b)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [self.lower, self.upper]
    else:
        raise(ValueError, 'ERROR: parameter type is undefined. Choose from Gaussian, Uniform, Gamma, Weibull, Cauchy, Exponential, TruncatedGaussian or Beta')

    return ab


# Recurrence coefficients for Jacobi type parameters
def jacobi_recurrence_coefficients(param_A, param_B, order):

    a0 = (param_B - param_A)/(param_A + param_B + 2.0)
    ab = np.zeros((int(order) + 1,2))
    b2a2 = param_B**2 - param_A**2

    if order > 0 :
        ab[0,0] = a0
        ab[0,1] = ( 2**(param_A + param_B + 1) * gamma(param_A + 1) * gamma(param_B + 1) )/( gamma(param_A + param_B + 2))

    for k in range(1,int(order) + 1):
        temp = k + 1
        ab[k,0] = b2a2/((2.0 * (temp - 1) + param_A + param_B) * (2.0 * temp + param_A + param_B))
        if(k == 1):
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B)) / ( (2 * (temp - 1) + param_A + param_B  )**2 * (2 * (temp - 1) + param_A + param_B + 1))
        else:
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B) * (temp - 1 + param_A + param_B) ) / ((2 * (temp - 1) + param_A + param_B)**2 * ( 2 *(temp -1) + param_A + param_B + 1) * (2 * (temp - 1) + param_A + param_B -1 ) )

    return ab

# Jacobi coefficients defined over [0,1]
def jacobi_recurrence_coefficients_01(param_A, param_B, order):

    ab = np.zeros((order+1,2))
    cd = jacobi_recurrence_coefficients(param_A, param_B, order)
    N = order + 1

    for i in range(0,int(N)):
        ab[i,0] = (1 + cd[i,0])/2.0

    ab[0,1] = cd[0,1]/(2**(param_A + param_B + 1))

    for i in range(1,int(N)):
        ab[i,1] = cd[i,1]/4.0

    return ab

# Recurrence coefficients for Hermite type parameters with a variance given by param_B + 0.5
# and mean of 0.0
def hermite_recurrence_coefficients(param_A, param_B, order):

    # Allocate memory
    ab = np.zeros((order,2))
    sigma2 = param_B

    if order == 1:
        ab[0,0] = 0
        ab[0,1] = gamma(param_A + 0.5)
        return ab

    # Adapted from Walter Gatuschi
    N = order - 1
    n = range(1,N+1)
    nh = [ k / 2.0 for k in n]
    for i in range(0,N,2):
        nh[i] = nh[i] + sigma2

    # Now fill in the entries of "ab"
    for i in range(0,order):
        if i == 0:
            ab[i,1] = gamma(sigma2 + 0.5)
        else:
            ab[i,1] = nh[i-1]
    ab[0,1] = gamma(param_A + 0.5)#2.0

    return ab


# Recurrence coefficients for Custom parameters
def custom_recurrence_coefficients(order, x, w):

    # Allocate memory for recurrence coefficients
    order = int(order)+1
    w = w / np.sum(w)
    ab = np.zeros((order+1,2))

    # Negate "zero" components
    nonzero_indices = []
    for i in range(0, len(x)):
        if w[i] != 0:
            nonzero_indices.append(i)

    ncap = len(nonzero_indices)
    x = x[nonzero_indices] # only keep entries at the non-zero indices!
    w = w[nonzero_indices]
    s = np.sum(w)

    temp = w/s
    ab[0,0] = np.dot(x, temp.T)
    ab[0,1] = s


    if order == 1:
        return ab

    p1 = np.zeros((1, ncap))
    p2 = np.ones((1, ncap))

    for j in range(0, order):
        p0 = p1
        p1 = p2
        p2 = ( x - ab[j,0] ) * p1 - ab[j,1] * p0
        p2_squared = p2**2
        s1 = np.dot(w, p2_squared.T)
        inner = w * p2_squared
        s2 = np.dot(x, inner.T)
        ab[j+1,0] = s2/s1
        ab[j+1,1] = s1/s
        s = s1

    return ab

# Compute the Jacobi matrix. The eigenvalues and eigenvectors of this matrix
# forms the basis of gaussian quadratures
def jacobiMatrix(self, order=None):

    if order is None:
        ab = recurrence_coefficients(self)
        order = self.order + 1
    else:
        ab = recurrence_coefficients(self, order)

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

# Computes 1D quadrature points and weights between [-1,1]
def getlocalquadrature(self, order=None, scale=None):

    # Check for extra input argument!
    if order is None:
        order = self.order + 1
    else:
        order = order + 1

    # Get the recurrence coefficients & the jacobi matrix
    recurrence_coeffs = recurrence_coefficients(self, order)
    JacobiMat = jacobiMatrix(self, order)

    # If statement to handle the case where order = 1
    if order == 1:

        # Check to see whether upper and lower bound are defined:
        if not self.lower or not self.upper:
            local_points = [computeMean(self)]
        else:
            local_points = [(self.upper - self.lower)/(2.0) + self.lower]
        local_weights = [1.0]
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
            w[u] = (V[0,i[u]]**2) # replace weights with right value
            p[u,0] = local_points[u]
        local_weights = w
        local_points = p

    if scale==True:
        scaled_points = p.copy()
        if self.param_type=='Uniform':
            for i in range(0, len(p)):
                scaled_points[i] = 0.5* ( p[i] + 1. ) * (self.upper - self.lower) + self.lower
        if self.param_type == 'Beta':
            for i in range(0, len(p)):
                scaled_points[i] =  p[i] * (self.upper - self.lower) + self.lower
        return scaled_points, local_weights
    else:
        # Return 1D gauss points and weights
        return local_points, local_weights

def jacobiEigenvectors(self, order=None):

    if order is None:
        order = self.order + 1

    JacobiMat = jacobiMatrix(self, order)
    if order == 1:
        V = [1.0]
    else:
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        V = V[:,i]

    return V


# Univariate orthogonal polynomial correspoding to the weight of the parameter
def orthoPolynomial_and_derivative(self, points, order=None):
    if order is None:
        order = self.order + 1
    else:
        order = order + 1
    gridPoints = np.asarray(points).copy()
    ab = recurrence_coefficients(self, order)
    if self.param_type == 'Uniform':
        if (gridPoints > 1.0).any() or (gridPoints < -1.0).any():
            raise(ValueError, "Points not normalized.")
    if self.param_type == 'Beta':
        if (gridPoints > 1.0).any() or (gridPoints < 0.0).any():
            raise(ValueError, "Points not normalized.")

    orthopoly = np.zeros((order, len(gridPoints))) # create a matrix full of zeros
    derivative_orthopoly = np.zeros((order, len(gridPoints)))

    # Convert the grid points to a numpy array -- simplfy life!
    gridPointsII = np.zeros((len(gridPoints), 1))
    for u in range(0, len(gridPoints)):
        gridPointsII[u,0] = gridPoints[u]
    orthopoly[0,:] = 1.0

    # Cases
    if order == 1: #CHANGED 2/2/18
        return orthopoly, derivative_orthopoly
    orthopoly[1,:] = ((gridPointsII[:,0] - ab[0,0]) * orthopoly[0,:] ) * (1.0)/(1.0 * np.sqrt(ab[1,1]) )
    derivative_orthopoly[1,:] = orthopoly[0,:] / (np.sqrt(ab[1,1]))
    if order == 2: #CHANGED 2/2/18
        return orthopoly, derivative_orthopoly

    if order >= 3: #CHANGED 2/2/18
        for u in range(2,order): #CHANGED 2/2/18
            # Three-term recurrence rule in action!
            orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0])*orthopoly[u-1,:]) - np.sqrt(ab[u-1,1])*orthopoly[u-2,:] )/(1.0 * np.sqrt(ab[u,1]))
        for u in range(2, order): #CHANGED 2/2/18
            # Four-term recurrence formula for derivatives of orthogonal polynomials!
            derivative_orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
        return orthopoly, derivative_orthopoly
