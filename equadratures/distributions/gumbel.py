"""The Gumbel distribution."""
import scipy.stats

from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import gumbel_r
RECURRENCE_PDF_SAMPLES = 50000
class Gumbel(Distribution):
    """
    The class defines a Gumbel object. It is the child of Distribution.

    :param int shape_parameter:
		The shape parameter associated with the Gumbel distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, location, scale_parameter, data):
        if location is None:
            if data is None:
                self.location = 0.0
                self.data = None
            else:
                self.data = data
                self.location = None
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
            params=scipy.stats.gumbel_r(self.data)
            self.location=params[0]
            self.scale_parameter=params[1]

        if self.scale_parameter < 0:
            raise ValueError('Invalid parameter in Gumbel distribution. Scale should be positive.')

        self.parent = gumbel_r(loc=self.location, scale=self.scale_parameter)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.location - 10.0, 20.0 + self.location, RECURRENCE_PDF_SAMPLES)

    def get_description(self):
        """
        A description of the Gumbel distribution.

        :param Gumbel self:
            An instance of the Gumbel class.
        :return:
            A string describing the Gumbel distribution.
        """
        text = "is a Gumbel distribution is characterised by its scale parameter, which here is"+str(self.scale_parameter)+" and its location, given by "+str(self.location)+"."
        return text
    def get_pdf(self, points=None):
        """
        A Gumbel probability density function.

        :param Gumbel self:
            An instance of the Gumbel class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Gumbel distribution.
        :return:
            Probability density values along the support of the Gumbel distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for get_pdf method')
    def get_cdf(self, points=None):
        """
        A Gumbel cumulative density function.

        :param Gumbel self:
            An instance of the Gumbel class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Gumbel distribution.
        :return:
            Cumulative density values along the support of the Gumbel distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for get_cdf method')
    def get_icdf(self, xx):
        """
        A Gumbel inverse cumulative density function.

        :param Gumbel:
            An instance of Gumbel class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Gumbel distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """
        Generates samples from the Gumbel distribution.

        :param Gumbel self:
            An instance of Gumbel class
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
