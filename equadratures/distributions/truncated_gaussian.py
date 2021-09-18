"""The Truncated Gaussian distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.gaussian import *
import numpy as np
from scipy.stats import truncnorm
RECURRENCE_PDF_SAMPLES = 8000

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
    :param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, mean, variance, lower, upper, data):
        if mean is None:
            if data is None:
                self.mean = 0.0
                self.data = None
            else:
                self.data = data
                self.mean = None
        else:
            self.mean = mean
            self.data = None
        if variance is None:
            if data is None:
                self.variance = 1.0
                self.data = None
            else:
                self.variance = None
                self.data = data
        else:
            self.variance = variance
            self.data = None

        if lower is None:
            if data is None:
                self.lower = -3.0
                self.data = None
            else:
                self.data = data
                self.lower = None
        else:
            self.lower = lower
            self.data = None

        if upper is None:
            if data is None:
                self.upper = 3.0
                self.data = None
            else:
                self.upper = None
                self.data = data
        else:
            self.upper = upper
            self.data = None

        if self.data is not None:
            params=truncnorm.fit(self.data)
            self.mean=params[2]
            self.variance=params[3]**2
            self.lower=(params[0]*params[3])+params[2]
            self.upper=(params[1]*params[3])+params[2]

        self.std = np.sqrt(self.variance)
        a = (self.lower - self.mean) / self.std
        b = (self.upper - self.mean) / self.std
        self.parent = truncnorm(a, b, loc=self.mean, scale=self.std)
        self.bounds = np.array([self.lower, self.upper])
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')

    def get_description(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = "a truncated Gaussian distribution with a center of "+str(self.mean)+" and a scale of "+str(self.std)+", and a lower bound of "+str(self.lower)+" and an upper bound of "+str(self.upper)+"."
        return text
    def get_pdf(self, points=None):
        """
        A truncated Gaussian probability distribution.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :param matrix points:
            Matrix of points over the support of the distribution; default value is 500.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Probability density values along the support of the truncated Gaussian distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getPDF method')
    def get_icdf(self, xx):
        """ A truncated gaussian inverse cumulative density function,
        :param truncnorm:
            An instance of Truncated-Gaussian class.
        :param array xx:
            A matrix of points at which the inverse of cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Truncated Gaussian distributuion.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """ Generates samples from the Truncated-Gaussian distribution.

         :param trunc-norm self:
             An instance of the Truncated-Gaussian class.
         :param integer m:
             Number of random samples. If no value is provided, a default of     5e5 is assumed.
         :return:
             A N-by-1 vector that contains the samples.
        """
        if m is not None:
           number = m
        else:
           number = 500000
        return self.parent.rvs(size=number)
    def get_cdf(self, points=None):
        """
        A truncated Gaussian cumulative density function.

        :param Exponential self:
            An instance of the TruncatedGaussian class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Cumulative density values along the support of the truncated Gaussian distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
