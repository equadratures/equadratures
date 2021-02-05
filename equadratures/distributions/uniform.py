""" Please add a file description here"""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import uniform
RECURRENCE_PDF_SAMPLES = 8000

class Uniform(Distribution):
    """
    The class defines a Uniform object. It is the child of Distribution.

    :param double mean:
		Mean of the Gaussian distribution.
	:param double variance:
		Variance of the Gaussian distribution.
    """
    def __init__(self, lower, upper):
        if lower is None:
            self.lower = 0.0
        else:
            self.lower = lower
        if upper is None:
            self.upper = 1.0
        else:
            self.upper = upper

        self.parent = uniform(loc=(self.lower), scale=(self.upper - self.lower))
        self.bounds = np.array([self.lower, self.upper])
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.shape_parameter_A = 0.
        self.shape_parameter_B = 0.
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)

    def get_description(self):
        """
        A description of the Gaussian.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            A string describing the Gaussian.
        """
        text = "is a uniform distribution over the support "+str(self.lower)+" to "+str(self.upper)+"."
        return text
    def get_cdf(self, points=None):
        """
        A uniform cumulative density function.
        :param points:
                Matrix of points which have to be evaluated
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the uniform distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_pdf(self, points=None):
        """
        A uniform probability distribution.
        :param points:
            Matrix of points which have to be evaluated
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the uniform distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for get_pdf method')
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the uniform distribution.

        :param Uniform self:
            An instance of the Uniform class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the uniform distribution.
        """
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper, order)
        return ab
    def get_icdf(self, xx):
        """
        A Uniform inverse cumulative density function.

        :param: Uniform self:
            An instance of Uniform class
        :param array xx:
            Points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Uniform distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m = None):
        """
        Generates samples from the Uniform distribution.

        :param: uniform self:
            An instance of Uniform class
        :param: integer m:
            NUmber of random samples. If no provided, a default number of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)
