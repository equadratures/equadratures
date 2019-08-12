"""The Truncated Gaussian distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.gaussian import *
import numpy as np
from scipy.stats import truncnorm
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

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
    """
    def __init__(self, mean, variance, lower, upper):
        if (mean is not None) and (variance is not None) and (lower is not None) and (upper is not None):
            meanParent = mean
            varianceParent = variance
            self.std = Gaussian(mean = 0.0, variance = 1.0)
            self.parent = Gaussian(mean = meanParent, variance = varianceParent)
            self.lower = lower
            self.upper = upper
            self.skewness = 0.0
            self.kurtosis = 0.0
            self.bounds = np.array([-np.inf, np.inf])
            self.beta  = (self.upper - self.parent.mean)/np.sqrt(self.parent.variance)
            self.alpha = (self.lower - meanParent)/np.sqrt(varianceParent)
            self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)

            self.parents = truncnorm(a =self.alpha , b =self.beta, loc=meanParent, scale=np.sqrt(varianceParent))
            self.mean = self.parents.mean()
            self.variance = self.parents.var()
            self.sigma = np.sqrt(self.variance)

    def get_description(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = "A truncated Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+", and a lower bound of "+str(self.lower)+" and an upper bound of "+str(self.upper)+"."
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
            return self.parents.pdf(points)
        else:
            raise(ValueError, 'Please digit an input for getPDF method')
    def get_icdf(self, xx):
        """ A truncated gaussian inverse cumulative density function,
        :param truncnorm:
            An instance of Truncated-Gaussian class.
        :param array xx:
            A matrix of points at which the inverse of cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Truncated Gaussian distributuion.
        """
        return self.parents.ppf(xx)
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
        return self.parents.rvs(size=number)
