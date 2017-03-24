#!/usr/bin/env python
"""Core class for setting the properties of a univariate parameter."""
import numpy as np
from scipy.special import gamma
import analyticaldistributions as analytical
import matplotlib.pyplot as plt
from plotting import parameterplot
class Parameter(object):
    
    """
    This class defines a univariate parameter. Below are details of its constructor.
    :param double lower: Lower bound for the parameter. 
    :param double upper: Upper bound for the parameter
    :param integer points: Number of quadrature points to be used for subsequent
         computations. 
    :param string param_type: The type of distribution that characteristizes the parameter. Options include:
            `Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, `TruncatedGaussian <https://en.wikipedia.org/wiki/Truncated_normal_distribution>`_, 
            `Beta <https://en.wikipedia.org/wiki/Beta_distribution>`_, `Cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_,
            `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_, `Uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_, 
            `Gamma <https://en.wikipedia.org/wiki/Gamma_distribution>`_ and `Weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_. 
            If no string is provided, a `Uniform` distribution is assumed.
    :param double shape_parameter_A: Most of the aforementioned distributions are characterized by two shape 
            parameters. For instance, in the case of a `Gaussian` (or `TruncatedGaussian`), this represents the mean. In the case of a Beta
            distribution this represents the alpha value. For a uniform distribution this input is not required.
    :param double shape_parameter_B: This is the second shape parameter that characterizes the distribution selected.
            In the case of a `Gaussian` or `TruncatedGaussian`, this is the variance. 
    :param boolean derivative_flag: If flag is set to 1, then derivatives are used in polynomial computations. The default value is set to 0.
    **Sample declarations** 
    ::
        # Uniform distribution with 5 points on [-2,2]
        >> Parameter(points=5, lower=-2, upper=2, param_type='Uniform')
        # Gaussian distribution with 3 points with mean=4.0, variance=2.5
        >> Parameter(points=3, shape_parameter_A=4, shape_parameter_B=2.5, 
        param_type='Gaussian') 
        # Gamma distribution with 15 points with k=1.0 and theta=2.0
        >> Parameter(points=15, shape_parameter_A=1.0, shape_parameter_B=2.0, 
        param_type='Gamma')
        # Exponential distribution with 12 points with lambda=0.5
        >> Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
    """

    # constructor
    def __init__(self, points, lower=None, upper=None, param_type=None, shape_parameter_A=None, shape_parameter_B=None, derivative_flag=None):

        # Check what lower and upper are...
        self.order = points
        
        if lower is None:
            self.lower = -1.0
        else:
            self.lower = lower
        
        if upper is None:
            self.upper = 1.0
        else:
            self.upper = upper

        if param_type is None:
            self.param_type = 'Uniform'
        else:
            self.param_type = param_type

        if shape_parameter_A is None:
            self.shape_parameter_A = 0
        else:
            self.shape_parameter_A = shape_parameter_A 

        if shape_parameter_B is None:
            self.shape_parameter_B = 0
        else:
            self.shape_parameter_B = shape_parameter_B 

        if derivative_flag is None:
            self.derivative_flag = 0 
        else:
            self.derivative_flag = derivative_flag  
        
        if self.param_type == 'TruncatedGaussian' :
            if upper is None or lower is None:
                raise(ValueError, 'parameter __init__: upper and lower bounds are required for a TruncatedGaussian distribution!')

         # Check that lower is indeed above upper
        if self.lower >= self.upper :
            raise(ValueError, 'parameter __init__: upper bounds must be greater than lower bounds!')
  
    # Routine for computing the mean of the distributions
    def computeMean(self):
        """
        Returns the mean of the parameter 
        :param Parameter self: An instance of the Parameter class
        :return: mu, mean of the parameter
        :rtype: double
        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> mu = var1.computeMean()
        """
        if self.param_type == "Gaussian":
            mu = self.shape_parameter_A
        elif self.param_type == "Exponential":
            mu = 1.0/self.shape_parameter_A
        elif self.param_type == "Cauchy":
            mu = self.shape_parameter_A # technically the mean is undefined!
        elif self.param_type == "Weibull":
            mu = self.shape_parameter_A * gamma(1.0 + 1.0/self.shape_parameter_B)
        elif self.param_type == "Gamma":
            mu = self.shape_parameter_A * self.shape_parameter_B
        return mu

    def plot(self, filename=None):
        N = 500
        x, y = self.getPDF(N)
        x2, y2 = self.getCDF(N)
        parameterplot(x, y, y2, filename, x_label='X', y_label1='PDF', y_label2='CDF')

    def getPDF(self, N):
        """
        Returns the probability density function of the parameter 
        :param Parameter self: An instance of the Parameter class
        :param integer N: Number of points along the x-axis 
        :return: x, 1-by-N matrix that contains the values of the x-axis along the support of the parameter 
        :rtype: ndarray
        :return: w, 1-by-N matrix that contains the values of the PDF of the parameter
        :rtype: ndarray
        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> x, y = var1.getPDF(50)
        """
        if self.param_type is "Gaussian":
            x, y = analytical.PDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.PDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper) 
        elif self.param_type is "Gamma":
            x, y = analytical.PDF_Gamma(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            x, y = analytical.PDF_WeibullDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            x, y = analytical.PDF_CauchyDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            x, y = analytical.PDF_UniformDistribution(N, self.lower, self.upper)
        elif self.param_type is "TruncatedGaussian":
            x, y = analytical.PDF_TruncatedGaussian(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            x, y = analytical.PDF_ExponentialDistribution(N, self.shape_parameter_A)
        else:
            raise(ValueError, 'parameter getPDF(): invalid parameter type!')
        return x, y

    def getSamples(self, m=None, graph=None):
        """
        Returns samples of the Parameter
        : param Parameter self: An instance of the Parameter class
        : param integer m: Number of random samples. If no value is provided, a default of 5e5 is assumed. 
        """
        if m is None:
            number_of_random_samples = 5e5
        else:
            number_of_random_samples = m
        
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.getiCDF(uniform_samples)

        return yy
    
    def getCDF(self, N):
        """
        Returns the cumulative density function of the parameter 
        :param Parameter self: An instance of the Parameter class
        :param integer N: Number of points along the x-axis 
        :return: x, 1-by-N matrix that contains the values of the x-axis along the support of the parameter 
        :rtype: ndarray
        :return: w, 1-by-N matrix that contains the values of the PDF of the parameter
        :rtype: ndarray
        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> x, y = var1.getCDF(50)
        """
        if self.param_type is "Gaussian":
            x, y = analytical.CDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.CDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper) 
        elif self.param_type is "Gamma":
            x, y = analytical.CDF_Gamma(N, self.shape_parameter_A, self.shape_parameter_B)
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
        Returns values of the inverse CDF
        : param Parameter self: An instance of the Parameter class
        : param numpy array x: 1-by-N array of doubles where each entry is between [0,1]
        : return: y, 1-by-N array where each entry is the inverse CDF of input x
        : rtype: ndarray
        **Notes**
        This routine is called by the getSamples function. It makes a call to analyticalDistributions
        """    
        if self.param_type is "Gaussian":
            y = analytical.iCDF_Gaussian(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            y = analytical.iCDF_BetaDistribution(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper) 
        elif self.param_type is "Gamma":
            y = analytical.iCDF_Gamma(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            y = analytical.iCDF_WeibullDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            y = analytical.iCDF_CauchyDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            y = x * 1.0
        elif self.param_type is "TruncatedGaussian":
            y = analytical.iCDF_TruncatedGaussian(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            y = analytical.iCDF_ExponentialDistribution(x, self.shape_parameter_A)
        else:
            raise(ValueError, 'parameter getiCDF(): invalid parameter type!')
        return y
    
    def getRecurrenceCoefficients(self, order=None):
        """
        Returns the recurrence coefficients of the parameter 
        :param Parameter self: An instance of the Parameter class
        :param int order: The number of recurrence coefficients required. By default this is
            the same as the number of points used when the parameter constructor is initiated.
        :return: ab, order-by-2 matrix that containts the recurrence coefficients
        :rtype: ndarray
        
        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, 
            param_type='Exponential')
            >> ab = getRecurrenceCoefficients()
        """

        return recurrence_coefficients(self, order)

    def getJacobiMatrix(self, order=None):
        """
        Returns the tridiagonal Jacobi matrix
        :param Parameter self: An instance of the Parameter class
        :param int order: The number of rows and columns of the JacobiMatrix that is required. By default, this 
            value is set to be the same as the number of points used when the parameter constructor is initiated.
        :return: J, order-by-order sized Jacobi tridiagonal matrix
        :rtype: ndarray
        **Sample declaration**
        :: 
            # Code to compute the first 5 quadrature points & weights
            >> var3 = Parameter(points=5, param_type='Beta', lower=0, upper=1, 
            shape_parameter_A=2, shape_parameter_B=3)
            >> J = var3.getJacobiMatrix()
        """
        return jacobiMatrix(self, order)

    def getJacobiEigenvectors(self, order=None):
        """
        Returns the eigenvectors of the tridiagonal Jacobi matrix. These are used for computing
        quadrature rules for numerical integration.
        :param Parameter self: An instance of the Parameter class
        :param int order: Number of eigenvectors required. This function makes the call getJacobiMatrix(order) and then computes
            the corresponding eigenvectors.
        :return: V, order-by-order matrix that contains the eigenvectors of the Jacobi matrix
        :rtype: ndarray
        **Sample declaration**
        :: 
            # Code to Jacobi eigenvectors
            >> var4 = Parameter(points=5, param_type='Gaussian', shape_parameter_A=0, shape_parameter_B=2)
            >> V = var4.getJacobiEigenvectors()
        """
        return jacobiEigenvectors(self, order)

    def getOrthoPoly(self, points, order=None):
        """
        Returns orthogonal polynomials & its derivatives, evaluated at a set of points.
        :param Parameter self: An instance of the Parameter class
        :param ndarray points: Points at which the orthogonal polynomial (and its derivatives) should be evaluated at
        :param int order: This value of order overwrites the order defined for the constructor.
        :return: orthopoly, order-by-k matrix where order defines the number of orthogonal polynomials that will be evaluated
            and k defines the points at which these points should be evaluated at.
        :rtype: ndarray
        :return: derivative_orthopoly, order-by-k matrix where order defines the number of derivative of the orthogonal polynomials that will be evaluated
            and k defines the points at which these points should be evaluated at.
        :rtype: ndarray
        **Sample declaration**
        :: 
            >> x = np.linspace(-1,1,10)
            >> var6 = Parameter(points=10, param_type='Uniform', lower=-1, upper=1)
            >> poly = var6.getOrthoPoly(x)
        """
        return orthoPolynomial_and_derivative(self, points, order)

    def getLocalQuadrature(self, order=None):
        """
        Returns the 1D quadrature points and weights for the parameter
        :param Parameter self: An instance of the Parameter class
        :param int N: Number of quadrature points and weights required. If order is not specified, then
            by default the method will return the number of points defined in the parameter itself.
        :return: points, N-by-1 matrix that contains the quadrature points
        :rtype: ndarray
        :return: weights, 1-by-N matrix that contains the quadrature weights
        :rtype: ndarray
        **Sample declaration**
        :: 
            # Code to compute the first 5 quadrature points & weights
            >> var1 = Parameter(points=5, shape_parameter_A=0.5, param_type='Exponential')
            >> p, w = var1.getLocalQuadrature()
        """
        return getlocalquadrature(self, order)

#-----------------------------------------------------------------------------------
#
#                               PRIVATE FUNCTIONS BELOW
#
#-----------------------------------------------------------------------------------
# Call different methods depending on the choice of the polynomial parameter
def recurrence_coefficients(self, order=None):

    # Preliminaries.
    N = 8000 # no. of points for analytical distributions.
    if order  is None:
        order = self.order

    # 1. Beta distribution
    if self.param_type is "Beta":
        #param_A = self.shape_parameter_B - 1 # bug fix @ 9/6/2016
        #param_B = self.shape_parameter_A - 1
        #if(param_B <= 0):
        #    error_function('ERROR: parameter_A (beta shape parameter A) must be greater than 1!')
        #if(param_A <= 0):
        #    error_function('ERROR: parameter_B (beta shape parameter B) must be greater than 1!')
        #ab =  jacobi_recurrence_coefficients_01(param_A, param_B , order)
        alpha = self.shape_parameter_A
        beta = self.shape_parameter_B
        lower = self.lower
        upper = self.upper
        x, w = analytical.PDF_BetaDistribution(N, alpha, beta, lower, upper)
        ab = custom_recurrence_coefficients(order, x, w)

    # 2. Uniform distribution
    elif self.param_type is "Uniform":
        self.shape_parameter_A = 0.0
        self.shape_parameter_B = 0.0
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, order)
        #lower = self.lower
        #upper = self.upper
        #x, w = analytical.UniformDistribution(N, lower, upper)
        #ab = custom_recurrence_coefficients(order, x, w)

    # 3. Analytical Gaussian defined on [-inf, inf]
    elif self.param_type is "Gaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        x, w  = analytical.PDF_GaussianDistribution(N, mu, sigma)
        ab = custom_recurrence_coefficients(order, x, w)

    # 4. Analytical Exponential defined on [0, inf]
    elif self.param_type is "Exponential":
        lambda_value = self.shape_parameter_A
        x, w  = analytical.PDF_ExponentialDistribution(N, lambda_value)
        ab = custom_recurrence_coefficients(order, x, w)

    # 5. Analytical Cauchy defined on [-inf, inf]
    elif self.param_type is "Cauchy":
        x0 = self.shape_parameter_A
        gammavalue = self.shape_parameter_B
        x, w  = analytical.PDF_CauchyDistribution(N, x0, gammavalue)
        ab = custom_recurrence_coefficients(order, x, w)

    # 5. Analytical Gamma defined on [0, inf]
    elif self.param_type is "Gamma":
        k = self.shape_parameter_A
        theta = self.shape_parameter_B
        x, w  = analytical.PDF_GammaDistribution(N, k, theta)
        ab = custom_recurrence_coefficients(order, x, w)

    # 6. Analytical Weibull defined on [0, inf]
    elif self.param_type is "Weibull":
        lambda_value= self.shape_parameter_A
        k = self.shape_parameter_B
        x, w  = analytical.PDF_WeibullDistribution(N, lambda_value, k)
        ab = custom_recurrence_coefficients(order, x, w)

    # 3. Analytical Truncated Gaussian defined on [a,b]
    elif self.param_type is "TruncatedGaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        a = self.lower
        b = self.upper
        x, w  = analytical.PDF_TruncatedGaussian(N, mu, sigma, a, b)
        ab = custom_recurrence_coefficients(order, x, w)

    else:
        error_function('ERROR: parameter type is undefined. Choose from Gaussian, Uniform, Gamma, Weibull, Cauchy, Exponential, TruncatedGaussian or Beta')

    return ab
    

# Recurrence coefficients for Jacobi type parameters
def jacobi_recurrence_coefficients(param_A, param_B, order):

    a0 = (param_B - param_A)/(param_A + param_B + 2.0)
    ab = np.zeros((int(order),2))
    b2a2 = param_B**2 - param_A**2

    if order > 0 :
        ab[0,0] = a0
        ab[0,1] = ( 2**(param_A + param_B + 1) * gamma(param_A + 1) * gamma(param_B + 1) )/( gamma(param_A + param_B + 2))

    for k in range(1,int(order)):
        temp = k + 1
        ab[k,0] = b2a2/((2.0 * (temp - 1) + param_A + param_B) * (2.0 * temp + param_A + param_B))
        if(k == 1):
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B)) / ( (2 * (temp - 1) + param_A + param_B  )**2 * (2 * (temp - 1) + param_A + param_B + 1))
        else:
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B) * (temp - 1 + param_A + param_B) ) / ((2 * (temp - 1) + param_A + param_B)**2 * ( 2 *(temp -1) + param_A + param_B + 1) * (2 * (temp - 1) + param_A + param_B -1 ) )

    return ab

# Jacobi coefficients defined over [0,1]
def jacobi_recurrence_coefficients_01(param_A, param_B, order):

    ab = np.zeros((order,2))
    cd = jacobi_recurrence_coefficients(param_A, param_B, order)
    N = order

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
    order = int(order)
    w = w / np.sum(w)
    ab = np.zeros((order,2))

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

    for j in range(0, order - 1):
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
        order = self.order
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
def getlocalquadrature(self, order=None):

    # Check for extra input argument!
    if order is None:
        order = self.order
   
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

        local_weights = recurrence_coeffs[0,1] * w  # normalizing step
        #local_weights =  w
        #if self.param_type == 'Uniform':
        #    local_weights = w * (self.upper - self.lower)
        local_points = p # re-label

    # Return 1D gauss points and weights
    return local_points, local_weights

def jacobiEigenvectors(self, order=None):

    if order is None:
        order = self.order

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
def orthoPolynomial_and_derivative(self, gridPoints, order=None):

    if order is None:
        order = self.order

    orthopoly = np.zeros((order, len(gridPoints))) # create a matrix full of zeros
    derivative_orthopoly = np.zeros((order, len(gridPoints)))

    ab = recurrence_coefficients(self, order)
    # Convert the grid points to a numpy array -- simplfy life!
    gridPointsII = np.zeros((len(gridPoints), 1))
    for u in range(0, len(gridPoints)):
        gridPointsII[u,0] = gridPoints[u]

    # First orthonormal polynomial is always 1
    orthopoly[0,:] = 1.0

    # Cases
    if order == 1:
        return orthopoly

    orthopoly[1,:] = ((gridPointsII[:,0] - ab[0,0]) * orthopoly[0,:] ) * (1.0)/(1.0 * np.sqrt(ab[1,1]) )

    if order == 2:
        return orthopoly

    if order >= 3:
        for u in range(2,order):
            # Three-term recurrence rule in action!
            orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0])*orthopoly[u-1,:]) - np.sqrt(ab[u-1,1])*orthopoly[u-2,:] )/(1.0 * np.sqrt(ab[u,1]))

    # Only if the derivative flag is on do we compute the derivative polynomial
    if self.derivative_flag == 1:
        if order == 1:
            return derivative_orthopoly

        derivative_orthopoly[1,:] = orthopoly[0,:] / (np.sqrt(ab[1,1]))

        if order == 2:
            return derivative_orthopoly

        if order >= 3:
            for u in range(2, order):
                # Four-term recurrence formula for derivatives of orthogonal polynomials!
                derivative_orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
        return orthopoly, derivative_orthopoly

    else:
        empty = np.mat([0])
        return orthopoly, empty
