"""The Truncated Gaussian distribution."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from distribution import Distribution

class TruncatedGaussian(Distribution):
    """
    The class defines a Truncated-Gaussian object. It is the child of Distribution.
    :param double mean:
		Mean of the truncated Gaussian distribution.
	:param double variance:
		Variance of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the truncated Gaussian distribution.
    """
    def __init__(self, mean=None, variance=None, lower=None, upper=None):
        self.mean = mean
        self.variance = variance
        self.lower = lower 
        self.upper = upper
        if self.variance is not None:   
            self.sigma = np.sqrt(self.variance)
        self.skewness = 0.0
        self.kurtosis = 0.0
        self.bounds = np.array([-np.inf, np.inf])

    def getDescription(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = "A truncated Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+", and a lower bound of "+str(self.lower)+" and an upper bound of "+str(self.upper)+"."
        return text

    def getPDF(self, N):
        """
        A truncated Gaussian probability distribution.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
		:param integer N:
            Number of equidistant points over the support of the distribution; default value is 500.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the truncated Gaussian distribution.
        """
        x = np.linspace(self.lower, self.upper, N)
        zeta = (x - self.mean) / (self.sigma)
        phi_zeta = 1.0 / np.sqrt(2 * np.pi)  * np.exp(-0.5 * zeta**2)
        alpha = (self.lower - self.mean) / (self.sigma)
        beta = (self.upper - self.mean) / (self.sigma)
        Z = 0.5*(1.0 + erf(beta/np.sqrt(2.0) )) -  0.5*(1.0 + erf(alpha/np.sqrt(2.0)  ))
        w = phi_zeta/(self.sigma * Z) 
        return x, w

    def getCDF(self, N):
        """
        A truncated Gaussian cumulative density function.

	    :param truncated Gaussian self:
            An instance of the Gaussian class.
        :param integer N:
            Number of points for defining the cumulative density function; default value is 500.
        :return:
            An array of N equidistant values over the support of the truncated Gaussian.
        :return:
            Gaussian cumulative density values.
        """
        def cumulative(x):
            return 0.5 * (1.0 + erf(x/np.sqrt(2.0)))
        x = np.linspace(self.lower, self.upper, N)
        zeta = (x - self.mean)/( self.sigma )
        alpha = (self.lower - self.mean)/( self.sigma )
        beta = (self.upper - self.mean)/( self.sigma )
        Z = cumulative(beta) - cumulative(alpha)
        w = (cumulative(zeta) - cumulative(alpha))/(Z)
        return x, w