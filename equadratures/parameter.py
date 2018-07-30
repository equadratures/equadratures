"Definition of a univariate parameter."
from distributions.gaussian import Gaussian
from distributions.uniform import Uniform
from distributions.chebyshev import Chebyshev
from distributions.beta import Beta
from distributions.cauchy import Cauchy
from distributions.exponential import Exponential
from distributions.gamma import Gamma
from distributions.weibull import Weibull
from distributions.rayleigh import Rayleigh
from distributions.chisquared import Chisquared
from distributions.truncated_gaussian import TruncatedGaussian
import numpy as np

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
        The type of distribution that characterizes the parameter. Options include: 
		`Chebyshev (arcsine) <https://en.wikipedia.org/wiki/Arcsine_distribution>`_, 
		`Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, 
		`Truncated-Gaussian <https://en.wikipedia.org/wiki/Truncated_normal_distribution>`_, 
		`Beta <https://en.wikipedia.org/wiki/Beta_distribution>`_, 
		`Cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_, 
		`Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_, 
		`Uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_, 
		`Gamma <https://en.wikipedia.org/wiki/Gamma_distribution>`_, 
		`Weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_. 
		If no string is provided, a `Uniform` distribution is assumed. If the user provides data, and would like to generate orthogonal polynomials (and quadrature rules) based on the data, they can set this option to be Custom.
    :param double shape_parameter_A:
        Most of the aforementioned distributions are characterized by two shape parameters. For instance, in the case of a `Gaussian` (or `TruncatedGaussian`), this represents the mean. In the case of a Beta distribution this represents the alpha value. For a uniform distribution this input is not required.
    :param double shape_parameter_B:
        This is the second shape parameter that characterizes the distribution selected. In the case of a `Gaussian` or `TruncatedGaussian`, this is the variance.
    :param data:
        A numpy array with data values (x-y column format). Note this option is only invoked if the user uses the Custom param_type.
    :param Endpoints:
        A boolean entry. If set to True, then the quadrature points and weights will have end-points.
    """
	def __init__(self, order, distribution, endpoints=False, shape_parameter_A=None, shape_parameter_B=None, lower=None, upper=None):
		self.name = distribution
		self.order = order 
		self.shape_parameter_A = shape_parameter_A
		self.shape_parameter_B = shape_parameter_B
		self.lower = lower
		self.upper = upper
		self.endpoints = endpoints
		self.setDistribution()
		self.setBounds()
		self.setMoments()

	def setDistribution(self):
		choices = {'gaussian': Gaussian(self.shape_parameter_A, self.shape_parameter_B),
			       'normal': Gaussian(self.shape_parameter_A, self.shape_parameter_B),
			       'uniform' : Uniform(self.lower, self.upper),
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
		distribution = choices.get(self.name.lower(), distributionError)
		self.distribution = distribution

	def setMoments(self):
		self.mean = self.distribution.mean 
		self.variance = self.distribution.variance
		
	def setBounds(self):
		self.bounds = self.distribution.bounds

	def getPDF(self, N=None, points=None):
		return self.distribution.getPDF(N,points)

	def getCDF(self, N=None, points=None):
		return self.distribution.getCDF(N,points)

	def getiCDF(self, xx):
		return self.distribution.getiCDF(xx)

	def getSamples(self, m):
		return self.distribution.getSamples(m)
	
	def getDescription(self):
		return self.distribution.getDescription()
	
	def getRecurrenceCoefficients(self, order=None):
		return self.distribution.getRecurrenceCoefficients(order)
		
	def getJacobiEigenVectors(self, order=None):
		if order is None:
			order = self.order + 1
			JacobiMat = self.jacobiMatrix(order)
			if order == 1:
				V = [1.0]
		else:
			D,V = np.linalg.eig(JacobiMat)
			V = np.mat(V) # convert to matrix
			i = np.argsort(D) # get the sorted indices
			i = np.array(i) # convert to array
			V = V[:,i]
		return V

	def jacobiMatrix(self, order=None):
		if order is None:
			ab = self.getRecurrenceCoefficients()
			order = self.order + 1
		else:
			ab = self.getRecurrenceCoefficients(order)

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

	def _getOrthoPoly(self, points, order=None):
		eps = 1e-15
		if order is None:
			order = self.order + 1
		else:
			order = order + 1
		gridPoints = np.asarray(points).copy()
		ab = self.getRecurrenceCoefficients(order)
		if ( any(gridPoints) < self.bounds[0]) or (any(gridPoints) > self.bounds[1] ) :
			for r in range(0, len(gridPoints)):
				gridPoints[r] = (gridPoints[r] - self.bounds[0])/(self.bounds[1] - self.bounds[0])

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
		
	def _getLocalQuadrature(self, order=None):
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
			return getlocalquadrature(self, order)
		elif self.endpoints is True:
			return getlocalquadraturelobatto(self, order)
		else:
			raise(ValueError, '_getLocalQuadrature:: Error with Endpoints entry!')

def getlocalquadrature(self, order=None):
    # Check for extra input argument!
    if order is None:
        order = self.order + 1
    else:
        order = order + 1

    # Get the recurrence coefficients & the jacobi matrix
    JacobiMat = self.jacobiMatrix(order)
    ab = self.getRecurrenceCoefficients(order+1)

    # If statement to handle the case where order = 1
    if order == 1:
        # Check to see whether upper and lower bound are defined:
        if not self.lower or not self.upper:
            p = self.distribution.mean
        else:
            p = [(self.upper - self.lower)/(2.0) + self.lower]
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

def getlocalquadraturelobatto(self, order=None):
    # Check for extra input argument!
    if order is None:
        order = self.order - 2
    else:
        order = order - 2
    a = self.distribution.shape_parameter_A
    b = self.distribution.shape_parameter_B
    N = order
    # Get the recurrence coefficients & the jacobi matrix
    ab = self.getRecurrenceCoefficients(order+2)
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

def distributionError():
	raise(ValueError, 'Please select a valid distribution for your parameter; documentation can be found at www.effective-quadratures.org') 
	
