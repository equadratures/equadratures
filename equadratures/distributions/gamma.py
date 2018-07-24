"""The Gamma distribution."""
import numpy as np
from distribution import Distribution
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

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

    def getPDF(self, N=None, points=None):
        """
        A gamma probability density function.
        
        :param Gamma self:
            An instance of the Gamma class.
        :param integer N:
            Number of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gamma distribution.
        """
        if N is not None:
            x = np.linspace(0, self.shape*self.scale*10, N)
            w = 1.0/(gamma(self.shape) * self.scale**self.shape ) * x**(self.shape - 1) * np.exp(-x /self.scale)
            return x, w
        elif points is not None:
             w = 1.0/(gamma(self.shape) * self.scale**self.shape ) * points**(self.shape - 1) * np.exp(-points /self.scale)
             return w
        else:
            raise(ValueError, 'Please digit an input for getPDF method')
    
    def getCDF(k, theta, N=None, points=None):
        """
        A gamma cumulative density function.
        
        :param Gamma self:
            An instance of the Gamma class.
        :param integer N:
            Number of points for defining the gamma cumulative density function.
        :return:
            An array of N equidistant values over the support of the gamma distribution.
        :return:
            Cumulative density values along the support of the gamma distribution.
        """
        if N is not None:
            x = np.linspace(0, self.shape* self.scale * 10.0 , N)
            w = 1.0/(gamma(k)) * gammainc(self.shape, x/self.scale)
            return x, w
        elif points is not None:
            w = 1.0/(gamma(k)) * gammainc(self.shape, points/self.scale)
            return w
        else:
            raise(ValueError, 'Please digit an input for getCDF method')

