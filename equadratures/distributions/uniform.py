import numpy as np
from distribution import Distribution
from recurrence_utils import jacobi_recurrence_coefficients
from scipy.stats import uniform
RECURRENCE_PDF_SAMPLES = 8000

class Uniform(Distribution):
    """
    The class defines a Uniform object. It is the child of Distribution.
    
    :param double mean:
		Mean of the Gaussian distribution.
	:param double variance:
		Variance of the Gaussian distribution.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.bounds = np.array([-1.0, 1.0])
        if (self.lower is not None) and (self.upper is not None):
            self.mean = 0.5 * (self.upper + self.lower)
            self.variance = 1.0/12.0 * (self.upper - self.lower)**2
            self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
            self.parent = uniform(loc=(self.lower), scale=(self.upper-self.lower))
	    
        self.skewness = 0.0
        self.shape_parameter_A = 0. 
        self.shape_parameter_B = 0.
	
    def getCDF(self, points=None):
        """
        A uniform cumulative density function.
        :param points: 
                Matrix of points which have to be evaluated 
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the uniform distribution.
        """
        if points is not None: 
            return self.parent.cdf(points)
        else:
            raise(ValueError, 'Please digit an input for getCDF method')

    def getPDF(self, points=None):
        """
        A uniform probability distribution.
        :param points:
            Matrix of points which have to be evaluated
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the uniform distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise(ValueError, 'Please digit an input for getPDF method')


    def getRecurrenceCoefficients(self, order):
        """
        Recurrence coefficients for the uniform distribution.
        
        :param Uniform self:
            An instance of the Uniform class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the uniform distribution.
        """
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper, order)
        return ab

    def getiCDF(self, xx):
        """
        A Uniform inverse cumulative density function.

        :param: Uniform self:
            An instance of Uniform class
        :param array xx:
            Points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Uniform distribution.
        """
        return self.parent.ppf(xx)

    def getSamples(self, m = None):
        """
        Generates samples from the Uniform distribution.

        :param: uniform self:
            An instance of Uniform class
        :param: integer m:
            NUmber of random samples. If no provided, a default number of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)

