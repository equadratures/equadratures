"""The Beta distribution."""

from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients

import numpy as np
from scipy.stats import beta
RECURRENCE_PDF_SAMPLES = 8000

class Beta(Distribution):
    """
    The class defines a Beta object. It is the child of Distribution.

    :param 
    """
    def __init__(self, **kwargs):
        first_arg = ['alpha', 'shape_parameter_A', 'shape_A']
        second_arg = ['beta', 'shape_parameter_B', 'shape_B']
        third_arg = ['lower', 'low', 'bottom']
        fourth_arg = ['upper','up', 'top']  
        self.name = 'beta'
        self.lower = None 
        self.upper = None
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.shape_A = value 
            if second_arg.__contains__(key):
                self.shape_B = value 
            if third_arg.__contains__(key):
                self.lower = value 
            if fourth_arg.__contains__(key):
                self.upper = value 
        if self.lower is None and self.upper is None:
            self.lower = -1. # Standard beta distribution defn'
            self.upper = 1.
        if self.lower is None or self.upper is None:
            raise ValueError('lower or upper bounds have not been specified!')
        if self.upper <= self.lower:
            raise ValueError('invalid beta distribution parameters: upper should be greater than lower.')
        if self.shape_A <= 0 or self.shape_B <= 0:
            raise ValueError('invalid beta distribution parameters: shape parameters must be positive!')
        loc = self.lower
        scale = self.upper - self.lower
        self.parent = beta(self.shape_A, self.shape_B, loc=loc, scale=scale)
        self.mean, self.variance, self.skewness, self.kurtosis = beta.stats(self.shape_A, self.shape_B, loc=loc,
                                                                            scale=scale, moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, lower=self.lower, upper=self.upper, mean=self.mean, variance=self.variance, skewness=self.skewness, kurtosis=self.kurtosis, x_range_for_pdf=self.x_range_for_pdf)
    def get_description(self):
        """
        A description of the beta distribution.

        :param Beta self:
            An instance of the beta class.
        :return:
            A string describing the beta distribution.
        """
        text = "is a beta distribution is defined over a support; given here as "+str(self.lower)+", to "+str(self.upper)+". It has two shape parameters, given here to be "+str(self.shape_A)+" and "+str(self.shape_B)+"."
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
            raise ValueError( 'Please specify an input for getPDF method')
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
            raise ValueError( 'Please digit an input for getCDF method')
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
        ab = jacobi_recurrence_coefficients(self.shape_A - 1.0
                                            , self.shape_B - 1.0
                                            , self.lower, self.upper, order)
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
