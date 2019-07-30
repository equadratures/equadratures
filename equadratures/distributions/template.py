"""The Distribution template."""

from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients

import numpy as np

PDF_SAMPLES = 500000

class Distribution(object):
    """
    The class defines a Distribution object. It serves as a template for all distributions.

    :param double lower:
        Lower bound of the support of the distribution.
    :param double upper:
        Upper bound of the support of the distribution.
    """
    def __init__(self, mean=None, variance=None, lower=None, upper=None, shape=None, scale=None, rate=None):
        self.mean = mean
        self.variance = variance
        self.lower = lower
        self.upper = upper
        self.rate = rate
        self.scale = scale
        self.x_range_for_pdf = []
    def get_description(self):
        """
        Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_pdf(self, points=None):
        """
        Returns the PDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_cdf(self, N=None, points=None):
        """
        Returns the CDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
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
        pass
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the distribution

        :param Distribution self:
            An instance of the distribution class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the distribution.
        """
        w_pdf = self.getPDF(self.x_range_for_pdf)
        ab = custom_recurrence_coefficients(self.x_range_for_pdf, w_pdf, order)
        return ab
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
        if m is None:
            number_of_random_samples = PDF_SAMPLES
        else:
            number_of_random_samples = m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.getiCDF(uniform_samples)
        return yy
