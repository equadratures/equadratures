"""The Gamma distribution."""
import numpy as np
from distribution import Distribution
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from scipy.stats import gamma
RECURRENCE_PDF_SAMPLES = 8000

class Gamma(Distribution):
    """
    The class defines a Gamma object. It is the child of Distribution.
    
    :param double shape:
		Shape parameter of the gamma distribution.
    :param double scale:
		Scale parameter of the gamma distribution.
    """
    def __init__(self, shape=None, scale=None):
        self.shape = shape
        self.scale = scale
        self.bounds = np.array([0.0, np.inf])
        if (self.shape is not None) and (self.scale is not None) and (self.shape > 0.0) : 
            self.mean = self.shape * self.scale
            self.variance = self.shape * self.scale**2
            self.skewness = 2.0 / np.sqrt(self.shape)
            self.kurtosis = 6.0 / self.shape # double-check!
            self.x_range_for_pdf = np.linspace(0, self.shape*self.scale*10, RECURRENCE_PDF_SAMPLES)
    
    def getDescription(self):
        """
        A description of the gamma distribution.
            
        :param Gamma self:
            An instance of the Gamma class.
        :return:
            A string describing the gamma distribution.
        """
        text = "A gamma distribution with a shape parameter of "+str(self.shape)+", and a scale parameter of "+str(self.scale)+"."
        return text

    def getPDF(self, points=None):
        """
        A gamma probability density function.
        
        :param Gamma self:
            An instance of the Gamma class.
        :param matrix points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gamma distribution.
        """
        if points is not None:
            return gamma.pdf(points, self.shape, loc=0.0, scale=self.scale)
        else:
            raise(ValueError, 'Please digit an input for getPDF method')
    
    def getCDF(self, points=None):
        """
        A gamma cumulative density function.
        
        :param Gamma self:
            An instance of the Gamma class.
        :param matrix points:
            Matrix of points for defining the gamma cumulative density function.
        :return:
            An array of N equidistant values over the support of the gamma distribution.
        :return:
            Cumulative density values along the support of the gamma distribution.
        """
        if points is not None:
            return gamma.cdf(points, self.shape, loc=0.0, scale=self.scale)
        else:
            raise(ValueError, 'Please digit an input for getCDF method')

    def getiCDF(self, xx):
        """
        A gamma inverse cumulative density function.
        
        :param gamma self:
            An instance of Gamma class.
        :param xx:
            An array of points at which the inverse of cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Gamma distribution.
        """
        return gamma.ppf(xx, self.shape, loc=0.0, scale=self.scale)

    def getSamples(self, m=None):
        """
         Generates samples from the Gamma distribution.
         
         :param Gamma self:
             An instance of the Gamma class.
         :param integer m:
             Number of random samples. If no value is provided, a default of     5e5 is assumed.
         :return:
             A N-by-1 vector that contains the samples.
        """
        if m is not None:
           number = m
        else:
           number = 500000
        return gamma.rvs(self.shape, loc=0.0, scale=self.scale, random_state=None, size = number)

