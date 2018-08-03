"""The Gaussian / Normal distribution."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from distribution import Distribution
import matplotlib.pyplot as plt
from scipy.stats import norm
RECURRENCE_PDF_SAMPLES = 8000

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
            self.x_range_for_pdf = np.linspace(-15.0 * self.sigma, 15.0*self.sigma, RECURRENCE_PDF_SAMPLES) + self.mean
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
        if m is not None:
            number = m
        else: 
            number = 500000
        return norm.rvs(self.mean, self.variance, size=number)

    def getPDF(self, points=None):
        """
        A Gaussian probability distribution.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gaussian distribution.
        """
        return norm.pdf(points, loc=self.mean, scale=self.variance )

    def getCDF(self, points=None):
        """
        A Gaussian cumulative density function.

	    :param Gaussian self:
            An instance of the Gaussian class.
        :param array points 
            Points for which the cumulative density function is required.
        :return:
            Gaussian cumulative density values.
        """
        return norm.cdf(points, loc=self.mean, scale=self.variance )

    def getiCDF(self, xx):
        """
        An inverse Gaussian cumulative density function.

        :param Gaussian self:
            An instance of the Gaussian class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the Gaussian distribution.
        """
        return norm.ppf(xx, loc=self.mean, scale=self.variance)
