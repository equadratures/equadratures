"""The Beta distribution."""

from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients

import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from scipy.stats import beta
RECURRENCE_PDF_SAMPLES = 8000

class Beta(Distribution):
    """
    The class defines a Beta object. It is the child of Distribution.

    :param double a:
        First shape parameter of the beta distribution. This value has to be greater than 0.
    :param double b:
            Second shape parameter of the beta distribution. This value has to be greater than 0.
    :param double lower:
        Lower bound of the support of the beta distribution.
    :param double upper:
        Upper bound of the support of the beta distribution.
    """
    def __init__(self, lower=None, upper=None, shape_A=None, shape_B=None):
        self.shape_A = shape_A
        self.shape_B = shape_B
        self.lower = lower
        self.upper = upper
        if (self.shape_A is not None) and (self.shape_B is not None):
            if self.shape_A >= 1. and self.shape_B >= 1.0:
                self.mean = (self.shape_A) / (self.shape_A + self.shape_B)
                self.variance = (self.shape_A * self.shape_B) / ( (self.shape_A + self.shape_B)**2 * (self.shape_A + self.shape_B + 1.0) )
                self.skewness = 2.0 * (self.shape_B - self.shape_A) * np.sqrt(self.shape_A + self.shape_B + 1.0) / ( (self.shape_A + self.shape_B + 2.0) * np.sqrt(self.shape_A * self.shape_B) )
                self.kurtosis = 6.0 * ((self.shape_A - self.shape_B)**2 * (self.shape_A + self.shape_B + 1.0) - self.shape_A * self.shape_B * (self.shape_A + self.shape_B + 2.0)  ) /( (self.shape_A * self.shape_B) * (self.shape_A + self.shape_B + 2.0) * (self.shape_A + self.shape_B + 3.0)) + 3.0
                self.bounds = np.array([0, 1])
                self.shape_parameter_A = self.shape_B - 1.0
                self.shape_parameter_B = self.shape_A - 1.0
                self.parent = beta(self.shape_A, self.shape_B)
        if (self.lower is not None) and (self.upper is not None):
            self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
    def get_description(self):
        """
        A description of the beta distribution.

        :param Beta self:
            An instance of the beta class.
        :return:
            A string describing the beta distribution.
        """
        text = "A beta distribution is defined over a support; given here as "+str(self.lower)+", to "+str(self.upper)+". It has two shape parameters, given here to be "+str(self.shape_A)+" and "+str(self.shape_B)+"."
        return text

    def get_pdf(self, points=None):
        """
        A beta probability density function.

        :param Beta self:
            An instance of the Beta class.
        :return:
            Probability density values along the support of the Beta distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise(ValueError, 'Please specify an input for getPDF method')

    def get_cdf(self, points=None):
        """
        A beta cumulative density function.

        :param Beta self:
            An instance of the Beta class.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the Gamma distribution.
        """
        if points is not None:
                return self.parent.cdf(points)
        else:
            raise(ValueError, 'Please digit an input for getCDF method')

    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the beta distribution.

        :param Beta self:
            An instance of the Beya class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the beta distribution.
        """
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper, order)
        return ab

    def get_icdf(self, xx):
        """
        A Beta inverse cumulative density function.

        :param Beta self:
            An instance of Beta class.
        :param array xx:
            Points at which the inverse cumulative density funcion needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Beta distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m =None):
        """ Generates samples from the Beta distribution.

            :param beta self:
                An instance of Beta class.
            :param integer m:
                Number of random samples. If no provided, a default value of 5e5 is assumed.
            :return:
                A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)
