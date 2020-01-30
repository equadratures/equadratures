"""The Lognormal distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import lognorm
RECURRENCE_PDF_SAMPLES = 50000
class Lognormal(Distribution):
    """
    The class defines a Lognormal object. It is the child of Distribution.

    :param int shape_parameter:
		The shape parameter associated with the Lognormal distribution.
    """
    def __init__(self, shape_parameter):
        self.shape_parameter = shape_parameter
        if self.shape_parameter is not None:
            self.bounds = np.array([0.0, np.inf])
            if self.shape_parameter > 0:
                mean, var, skew, kurt = lognorm.stats(s=self.shape_parameter, moments='mvsk')
                self.parent = lognorm(s=self.shape_parameter)
                self.mean = mean
                self.variance = var
                self.skewness = skew
                self.kurtosis = kurt
                self.x_range_for_pdf = np.linspace(0., 20., RECURRENCE_PDF_SAMPLES)
    def get_description(self):
        """
        A description of the Lognormal distribution.

        :param Lognormal self:
            An instance of the Lognormal class.
        :return:
            A string describing the Lognormal distribution.
        """
        text = "is a Lognormal distribution is characterised by its shape parameter, which here is"+str(self.shape_parameter)+"."
        return text
    def get_pdf(self, points=None):
        """
        A Lognormal probability density function.

        :param Lognormal self:
            An instance of the Logistic class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Lognormal distribution.
        :return:
            Probability density values along the support of the Lognormal distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for get_pdf method')
    def get_cdf(self, points=None):
        """
        A Lognormal cumulative density function.

        :param Lognormal self:
            An instance of the Logistic class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Lognormal distribution.
        :return:
            Cumulative density values along the support of the Lognormal distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for get_cdf method')
    def get_icdf(self, xx):
        """
        A Lognormal inverse cumulative density function.

        :param Gumbel:
            An instance of Logistic class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Lognormal distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """
        Generates samples from the Lognormal distribution.

        :param Logistic self:
            An instance of Lognormal class
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
