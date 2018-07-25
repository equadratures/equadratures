"""The Gaussian / Normal distribution."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from distribution import Distribution
import matplotlib.pyplot as plt
class Gaussian(Distribution):
    """
    The class defines a Gaussian object. It is the child of Distribution.

    :param double mean:
		Mean of the Gaussian distribution.
	:param double variance:
		Variance of the Gaussian distribution.
    """
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        if self.variance is not None:
            self.sigma = np.sqrt(self.variance)
        self.skewness = 0.0
        self.kurtosis = 0.0
        self.bounds = np.array([-np.inf, np.inf])

    def getDescription(self):
        """
        A description of the Gaussian.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            A string describing the Gaussian.
        """
        text = "A Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+"."
        return text

    def getSamples(self, m=None):
        """
        Generates samples from the Gaussian distribution.
        :param Gaussian self:
            An instance of the Gaussian class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is None:
            number_of_random_samples = PDF_SAMPLES
        else:
            number_of_random_samples = m
        return np.random.randn(number_of_random_samples, 1)*self.sigma + self.mean

    def getPDF(self, N=None, points=None):
        """
        A Gaussian probability distribution.

        :param Gaussian self:
            An instance of the Gaussian class.
		:param integer N:
            Number of equidistant points over the support of the distribution; default value is 500.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gaussian distribution.
        """
        if N is not None:
            x = np.linspace(-15*self.sigma, 15*self.sigma, N)
            x = x + self.mean 
            w = 1.0/( np.sqrt(2 * self.variance * np.pi) ) * np.exp(-(x - self.mean)**2 * 1.0/(2 * self.variance) )
            return x, w
        elif points is not None:
             w = 1.0/( np.sqrt(2 * self.variance * np.pi) ) * np.exp(-(points - self.mean)**2 * 1.0/(2 * self.variance) )
             return w
        else:
             raise(ValueError, 'Please digit an input for getPDF method')

    def getCDF(self, N=None, points=None):
        """
        A Gaussian cumulative density function.

	    :param Gaussian self:
            An instance of the Gaussian class.
        :param integer N:
            Number of points for defining the cumulative density function; default value is 500.
        :param array points 
            Points for which the cumulative density function is required.
        :return:
            An array of N equidistant values over the support of the Gaussian.
        :return:
            Gaussian cumulative density values.
        """
        if N is not None:
            x = np.linspace(-15*self.sigma, 15*self.sigma, N)
            x = x + self.mean # scaling it by the mean!
            w = 0.5*(1 + erf((x - self.mean)/(self.sigma * np.sqrt(2) ) ) )
            return x, w
        elif points is not None:
            w = 0.5*(1+erf((points-self.mean)/(self.sigma*np.sqrt(2))))
            return w
        else:
            raise(ValueError, 'getCDF(): Please check your input arguments!')

    def getiCDF(self, xx):
        """
        An inverse Gaussian cumulative density function.

        :param Gaussian self:
            An instance of the Gaussian class.
        :param array points:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the Gaussian distribution.
        """
        return self.mean + np.sqrt(self.variance) * np.sqrt(2.0) * erfinv(2.0*xx - 1.0)
