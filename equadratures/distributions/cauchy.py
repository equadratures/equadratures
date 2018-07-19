"""The Cauchy distribution."""
import numpy as np
from distribution import Distribution
from recurrence_utils import custom_recurrence_coefficients

class Cauchy(Distribution):
    """
    The class defines a Cauchy object. It is the child of Distribution.
    
    :param double location:
		Location parameter of the Cauchy distribution.
    :param double scale:
		Scale parameter of the Cauchy distribution.
    """
    def __init__(self, location=None, scale=None):
        self.location = location
        self.scale = scale
        self.bounds = np.array([-np.inf, np.inf])
        self.mean = np.nan
        self.variance = np.nan
        self.skewness = np.nan
        self.kurtosis = np.nan
    
    def getDescription(self):
        """
        A description of the Cauchy distribution.
        
        :param Cauchy self:
            An instance of the Cauchy class.
        :return:
            A string describing the Cauchy distribution.
        """
        text = "A Cauchy distribution has an undefined mean and variance; its location parameter is "+str(self.location)+", and its scale parameter is "+str(self.scale)+"."
        return text
    
    def getPDF(self, N):
        """
        A Cauchy probability density function.
        
        :param Cauchy self:
            An instance of the Cauchy class.
        :param int N:
            Number of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Cauchy distribution.
        """
        x = np.linspace(-15*self.scale, 15*self.scale, N)
        x = x + self.location
        w = 1.0/(np.pi * self.scale * (1 + ((x - self.location)/(self.scale))**2) )
        return x, w

    def getCDF(self, N):
        """
        A Cauchy cumulative density function.
        
        :param Cauchy self:
            An instance of the Cauchy class.
        :param integer N:
            Number of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the Cauchy distribution.
        """
        x = np.linspace(-15*self.scale, 15*self.scale, N)
        x = x + self.location
        w = 1.0/np.pi * np.arctan((x - self.location) / self.scale) + 0.5
        return x, w

    def getiCDF(self, xx):
        """
        An inverse Cauchy cumulative density function.
        
        :param Cauchy self:
            An instance of the Cauchy class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0, 1].
        :return:
            Inverse CDF samples associated with the Cauchy distribution.
        """
        return self.location + self.scale * np.tan(np.pi * (xx - 0.5))

