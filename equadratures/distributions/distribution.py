"""The Distribution template."""
import numpy as np
from recurrence_utils import custom_recurrence_coefficients
RECURRENCE_PDF_SAMPLES = 8000
PDF_SAMPLES = 500000

class Distribution(object):
    """
    The class defines a Distribution object. It serves as a template for all distributions.
        
    :param double lower:
        Lower bound of the support of the Chebyshev (arcsine) distribution.
    :param double upper:
        Upper bound of the support of the Chebyshev (arcsine) distribution.
    """
    def __init__(self, mean=None, variance=None, lower=None, upper=None, shape=None, scale=None, rate=None):
        self.mean = mean
        self.variance = variance
        self.lower = lower
        self.upper = upper
        self.rate = rate
        self.scale = scale
        
    
    def getDescription(self):
        pass
        
	def getPDF(self, N):
		pass

	def getCDF(self, N):
		pass

    def getiCDF(self, xx):
        """
        An inverse cumulative density function.
    
        :param Distribution self:
                An instance of the distribution class.
        :param xx:
                A numpy array of uniformly distributed samples between [0,1].
        :return:
                Inverse CDF samples associated with the gamma distribution.
        """
        yy = []
        [x, c] = self.getCDF(1000)
        for k in range(0, len(xx)):
                for i in range(0, len(x)):
                    if ( (xx[k]>=c[i]) and (xx[k]<=c[i+1]) ):
                        value =  float( (xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i] )
                        yy.append(value)
                        break
        return yy

    def getRecurrenceCoefficients(self, order):
        """
        Recurrence coefficients for the distribution
        
        :param Distribution self:
            An instance of the distribution class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the distribution.
        """
        x, w  = self.getPDF(RECURRENCE_PDF_SAMPLES)
        ab = custom_recurrence_coefficients(x, w, order)
        return ab

    def getSamples(self, m=None):
        """
        Generates samples from the distribution.
            
        :param Distribution self:
            An instance of the distribution class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is None:
            number_of_random_samples = PDF_SAMPLES
        else:
            number_of_random_samples = m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.getiCDF(uniform_samples)
        return yy

		
