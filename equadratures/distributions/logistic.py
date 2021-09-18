"""The Logistic distribution."""
import scipy.stats

from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import logistic
RECURRENCE_PDF_SAMPLES = 50000
class Logistic(Distribution):
    """
    The class defines a Logistic object. It is the child of Distribution.

    :param int shape_parameter:
		The shape parameter associated with the Logistic distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, location, scale_parameter, data):
        if location is None:
            if data is None:
                self.location = 0.0
                self.data = None
            else:
                self.location = None
                self.data = data
        else:
            self.location = location
            self.data = None
        if scale_parameter is None:
            if data is None:
                self.scale_parameter = 1.0
                self.data = None
            else:
                self.scale_parameter = None
                self.data = data
        else:
            self.scale_parameter = scale_parameter
            self.data = None

        self.bounds = np.array([-np.inf, np.inf])

        if self.data is not None:
            params=scipy.stats.logistic(self.data)
            self.location = params[0]
            self.scale_parameter = params[1]

        if self.scale_parameter < 0:
            raise ValueError('Invalid parameter in Logistic distribution. Scale should be positive.')

        self.parent = logistic(loc=self.location, scale=self.scale_parameter)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.location - 10.0, 20.0 + self.location, RECURRENCE_PDF_SAMPLES)

    def get_description(self):
        """
        A description of the Logistic distribution.

        :param Logistic self:
            An instance of the Logistic class.
        :return:
            A string describing the Logistic distribution.
        """
        text = "is a Logistic distribution is characterised by its scale parameter, which here is"+str(self.scale_parameter)+" and its location, given by "+str(self.location)+"."
        return text
    def get_pdf(self, points=None):
        """
        A Logistic probability density function.

        :param Logistic self:
            An instance of the Logistic class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Logistic distribution.
        :return:
            Probability density values along the support of the Logistic distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for get_pdf method')
    def get_cdf(self, points=None):
        """
        A Logistic cumulative density function.

        :param Logistic self:
            An instance of the Logistic class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Logistic distribution.
        :return:
            Cumulative density values along the support of the Logistic distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for get_cdf method')
    def get_icdf(self, xx):
        """
        A Logistic inverse cumulative density function.

        :param Gumbel:
            An instance of Logistic class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Logistic distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """
        Generates samples from the Logistic distribution.

        :param Logistic self:
            An instance of Logistic class
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e05 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)
