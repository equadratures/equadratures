"""The Rayleigh distribution."""
import numpy as np
from distribution import Distribution

class Rayleigh(Distribution):
    """
    The class defines a Rayleigh object. It is the child of Distribution.
    
    :param double scale:
		Scale parameter of the Rayleigh distribution.
    """
    def __init__(self, scale):
        self.scale = scale
        self.bounds = np.array([0.0, np.inf])
        if self.scale > 0:
            self.mean = self.scale * np.sqrt(np.pi / 2.0)
            self.variance = self.scale**2 * (4.0 - np.pi)/ 2.0 
            self.skewness = 2.0 * np.sqrt(np.pi) * (np.pi - 3.0) / ((4.0 - np.pi)**(1.5))
            self.kurtosis = -(6 * np.pi**2 - 24 * np.pi + 16.0 )/( (4 - np.pi)**(1.5)) + 3.0 

    def getDescription(self):
        """
        A description of the Rayleigh distribution.
            
        :param Rayleigh self:
            An instance of the Rayleigh class.
        :return:
            A string describing the Rayleigh distribution.
        """
        text = "A Rayleigh distribution is characterised by its scale parameter, which is"+str(self.scale)+"."
        return text

    def getPDF(self, N):
        """
        A Rayleigh probability density function.
        
        :param Rayleigh self:
            An instance of the Rayleigh class.
        :param integer N:
            Number of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Rayleigh distribution.
        :return:
            Probability density values along the support of the Rayleigh distribution.
        """
        xreal = np.linspace(0.0, 3.5*self.scale)
        wreal = xreal / (self.scale**2) * np.exp( - xreal**2 * 1.0/(2.0 * self.scale**2 ))
        return xreal, wreal

    def getCDF(self, N):
        """
        A Rayleigh cumulative density function.
        
        :param Rayleigh self:
            An instance of the Rayleigh class.
        :param integer N:
            Number of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Rayleigh distribution.
        :return:
            Cumulative density values along the support of the Rayleigh distribution.
        """
        xreal = np.linspace(0.0, 3.5*self.scale)
        wreal = 1.0 - np.exp(-xreal**2 * 1.0/(2.0 * self.scale**2))
        return xreal, wreal