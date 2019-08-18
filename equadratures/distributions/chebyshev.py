"""The Chebyshev / Arcsine distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import arcsine
RECURRENCE_PDF_SAMPLES = 8000
class Chebyshev(Distribution):
    """
    The class defines a Chebyshev object. It is the child of Distribution.

    :param double lower:
		Lower bound of the support of the Chebyshev (arcsine) distribution.
	:param double upper:
		Upper bound of the support of the Chebyshev (arcsine) distribution.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.bounds = np.array([0.0, 1.0])
        if ( self.lower is not None ) and (self.upper is not None) :
         	self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        self.mean = 0.5
        self.variance = 1.0/8.0
        self.skewness = 0.0
        self.shape_parameter_A = -0.5
        self.shape_parameter_B = -0.5
    def get_description(self):
        """
        A description of the Chebyshev (arcsine) distribution.

        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :return:
            A string describing the Chebyshev (arcsine) distribution.
        """
        text = "A Chebyshev (arcsine) distribution is characterised by its lower bound, which is"+str(self.lower)+" and its upper bound, which is"+str(self.upper)+"."
        return text
    def get_pdf(self, points=None):
        """
        A Chebyshev probability density function.

        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N the support of the Chebyshev (arcsine) distribution.
        :return:
            Probability density values along the support of the Chebyshev (arcsine) distribution.
        """
        if points is not None:
            return arcsine.pdf(points)
        else:
            raise(ValueError, 'Please digit an input for getPDF method')
    def get_cdf(self, points=None):
        """
        A Chebyshev cumulative density function.

        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N values over the support of the Chebyshev (arcsine) distribution.
        :return:
            Cumulative density values along the support of the Chebyshev (arcsine) distribution.
        """
        if points is not None:
             return arcsine.cdf(points)
        else:
            raise(ValueError, 'Please digit an input for getCDF method')
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the Chebyshev distribution.

        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the Chebyshev distribution.
        """
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper, order)
        return ab
    def get_icdf(self, xx):
        """
        A Arcisine inverse cumulative density function.

        :param Arcsine self:
            An instance of Arcisine class.
        :param xx:
            A matrix of points at which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Arcisine distribution.
        """
        return arcsine.ppf(xx)
    def get_samples(self, m=None):
        """
        Generates samples from the Arcsine distribution.

        :param arcsine self:
            An instance of Arcsine class.
        :param integer m:
            Number of random samples. If not provided, a default of 5e05 is assumed.

        """
        if m is not None:
            number = m
        else:
            number = 500000
        return arcsine.rvs(size=number)
