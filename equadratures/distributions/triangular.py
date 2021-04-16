"""The Triangular distrubution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
from scipy.stats import triang
RECURRENCE_PDF_SAMPLES = 8000

class Triangular(Distribution):
    """
    The class defines a Triangular object.

    :param double lower:
        Lower bound of the support of the distribution.
    :param double upper:
        Upper bound of the support of the distribution.
    :param double mode:
        Mode of the distribution.
    """
    def __init__(self, lower=None, upper=None, mode=None):
        self.lower = lower # loc
        self.upper = upper
        self.mode = mode
        self.bounds = np.array([0, 1.0])
        self.scale = upper - lower # scale
        self.shape = (self.mode - self.lower) / (self.upper - self.lower) # c

        
        if (self.lower is not None) and (self.upper is not None) and (self.mode is not None) :
            mean, var, skew, kurt = triang.stats(c=self.shape, loc=self.lower, scale=self.scale, moments='mvsk')
            self.mean = mean
            self.variance = var
            self.skewness = skew
            self.kurtosis = kurt
            self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
            self.parent = triang(loc=self.lower, scale=self.scale, c=self.shape)


    def get_description(self):
        """
        Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        text = "is a triangular distribution with a mode of "+str(self.mode)+" over the support "+str(self.lower)+" to "+str(self.upper)+"."
        return text
    def get_cdf(self, points=None):
        """
        Returns the CDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')

    def get_pdf(self, points=None):
        """
        Returns the PDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for get_pdf method')
    def get_icdf(self, xx):
        """
        An inverse cumulative density function.

        :param Distribution self:
                An instance of the distribution class.
        :param xx:
                A numpy array of uniformly distributed samples between [0,1].
        :return:
                Inverse CDF samples associated with the gamma distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m=None):
        """
        Generates samples from the distribution.

        :param Distribution self:
            An instance of the distribution class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)