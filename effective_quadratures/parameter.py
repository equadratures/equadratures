"""Core class for setting the properties of a univariate parameter."""
import numpy as np
from scipy.special import gamma
import analyticaldistributions as analytical
from utils import error_function
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
            In the case of a `Gaussian` or `TruncatedGaussian` this is the variance. 
    :param derivative_flag: If flag is 
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
    def __init__(self, points, lower=None, upper=None, param_type = None, shape_parameter_A=None, shape_parameter_B=None, derivative_flag=None):
     
        # Check that lower is indeed above upper
        if lower >= upper :
            error_function('Parameter: upper must be larger than lower')
            
        self.lower = lower # double
        self.upper = upper # double
        self.order = points # integer

        # Check what lower and upper are...
        if lower is None:
            self

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
        
    def getPDF(self, N):
        """
        Returns the PDF of the parameter 

        :param ndarray x: N-by-n matrix of points in the space of active
            variables.
        :param int N: merely there satisfy the interface of `regularize_z`. It
            should not be anything other than 1.

        :return: x, N-by-(m-n)-by-1 matrix that contains a value of the inactive
            variables for each value of the inactive variables.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> x, y = var1.getPDF()
        """
        return 0

    def getCFD(self):
        """
        Returns the CDF of the parameter 

        :param ndarray x: N-by-n matrix of points in the space of active
            variables.
        :param int N: merely there satisfy the interface of `regularize_z`. It
            should not be anything other than 1.

        :return: x, N-by-(m-n)-by-1 matrix that contains a value of the inactive
            variables for each value of the inactive variables.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> x, y = var1.getCDF()
        """
        return 0
          
    def getDerivativeFlag(self):
        return self.derivative_flag

    def getOrder(self):
        return self.order

    def getParamType(self):
        return self.param_type

    def getLowerBound(self):
        return self.lower_bound

    def getUpperBound(self):
        return self.upper_bound

    def getShapeParameterA(self):
        return self.shape_parameter_A

    def getShapeParameterB(self):
        return self.shape_parameter_B

    def getRecurrenceCoefficients(self, *argv):
        """
        Returns the recurrence coefficients of the parameter 

        :param ndarray x: N-by-n matrix of points in the space of active
            variables.
        :param int N: merely there satisfy the interface of `regularize_z`. It
            should not be anything other than 1.

        :return: x, N-by-(m-n)-by-1 matrix that contains a value of the inactive
            variables for each value of the inactive variables.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
            >> x, y = var1.getPDF()
        """

        return recurrence_coefficients(self, argv)

    def getJacobiMatrix(self, *argv):
        return jacobiMatrix(self, *argv)

    def getJacobiEigenvectors(self, *argv):
        return jacobiEigenvectors(self, *argv)

    def getOrthoPoly(self, points, *argv):
        return orthoPolynomial_and_derivative(self, points, *argv)

    def getLocalQuadrature(self, order):
        return getlocalquadrature(self, order)


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
            error_function('ERROR: parameter_A (beta shape parameter A) must be greater than 1!')
        if(param_A <= 0):
            error_function('ERROR: parameter_B (beta shape parameter B) must be greater than 1!')
        ab =  jacobi_recurrence_coefficients_01(param_A, param_B , order)

    # 2. Uniform distribution
    elif self.param_type is "Uniform":
        self.shape_parameter_A = 0.0
        self.shape_parameter_B = 0.0
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, order)

    # 3. Analytical Gaussian defined on [-inf, inf]
    elif self.param_type is "Gaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        x, w  = analytical.Gaussian(mu, sigma, N)
        ab = custom_recurrence_coefficients(order, x, w)

    # 4. Analytical Exponential defined on [0, inf]
    elif self.param_type is "Exponential":
        lambda_value = self.shape_parameter_A
        x, w  = analytical.ExponentialDistribution(N, lambda_value)
        ab = custom_recurrence_coefficients(order, x, w)

    # 5. Analytical Cauchy defined on [-inf, inf]
    elif self.param_type is "Cauchy":
        x0 = self.shape_parameter_A
        gammavalue = self.shape_parameter_B
        x, w  = analytical.CauchyDistribution(N, x0, gammavalue)
        ab = custom_recurrence_coefficients(order, x, w)

    # 5. Analytical Gamma defined on [0, inf]
    elif self.param_type is "Gamma":
        k = self.shape_parameter_A
        theta = self.shape_parameter_B
        x, w  = analytical.GammaDistribution(N, k, theta)
        ab = custom_recurrence_coefficients(order, x, w)

    # 6. Analytical Weibull defined on [0, inf]
    elif self.param_type is "Weibull":
        lambda_value= self.shape_parameter_A
        k = self.shape_parameter_B
        x, w  = analytical.WeibullDistribution(N, lambda_value, k)
        ab = custom_recurrence_coefficients(order, x, w)

    # 3. Analytical Truncated Gaussian defined on [a,b]
    elif self.param_type is "TruncatedGaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        a = self.lower_bound
        b = self.upper_bound
        x, w  = analytical.TruncatedGaussian(N, mu, sigma, a, b)
        ab = custom_recurrence_coefficients(order, x, w)

    #elif self.param_type == "Gaussian" or self.param_type == "Normal":
    #    param_B = self.shape_parameter_B - 0.5
    #   if(param_B <= -0.5):
    #        utils.error_function('ERROR: parameter_B (variance) must be greater than 0')
    #    else:
    #        ab =  hermite_recurrence_coefficients(self.shape_parameter_A, param_B, order)
    else:
        error_function('ERROR: parameter type is undefined. Choose from Gaussian, Uniform, Gamma, Weibull, Cauchy, Exponential, TruncatedGaussian or Beta')

    return ab

# Recurrence coefficients for Jacobi type parameters
def jacobi_recurrence_coefficients(param_A, param_B, order):

    a0 = (param_B - param_A)/(param_A + param_B + 2.0)
    ab = np.zeros((order,2))
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
def jacobiMatrix(self, order_to_use=None):

    if order_to_use is None:
        ab = recurrence_coefficients(self)
        order = self.order
    else:
        order = order_to_use
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
def getlocalquadrature(self, order_to_use=None):

    # Check for extra input argument!
    if order_to_use is None:
        order = self.order
    else:
        order = order_to_use

    # Get the recurrence coefficients & the jacobi matrix
    recurrence_coeffs = recurrence_coefficients(self, order)
    JacobiMat = jacobiMatrix(self, order)

    # If statement to handle the case where order = 1
    if order == 1:

        # Check to see whether upper and lower bound are defined:
        if not self.lower_bound or not self.upper_bound:
            local_points = [computeMean(self)]
        else:
            local_points = [(self.upper_bound - self.lower_bound)/(2.0) + self.lower_bound]
        local_weights = [1.0]
    else:
        # Compute eigenvalues & eigenvectors of Jacobi matrix
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        local_points = np.sort(D) # sort by the eigenvalues
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        w = np.linspace(1,order+1,order) # create space for weights
        p = np.ones((order,1))
        for u in range(0, len(i) ):
            w[u] = (V[0,i[u]]**2) # replace weights with right value
            p[u,0] = local_points[u]

        local_weights = recurrence_coeffs[0,1] * w  # normalizing step
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

# This routine computes the mean of the distribution, depending on which distribution
# is selected. This function is called by getlocalquadratures()
def computeMean(self):
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

    # Zeroth order
    # original!
    # orthopoly[0,:] = (1.0)/(1.0 * np.sqrt(ab[0,1]) ) # Correct!
    # New
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
