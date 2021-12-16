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
    def __init__(self, **kwargs):
        first_arg = ['lower', 'low', 'bottom']
        second_arg = ['upper','up', 'top']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        self.name = 'uniform'
        self.lower = None 
        self.upper = None 
        self.order = 2
        self.endpoints = 'none'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.lower = value
            if second_arg.__contains__(key):
                self.upper = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
        if self.lower is None or self.upper is None:
            raise ValueError('lower or upper bounds have not been specified!')
        if self.upper <= self.lower:
            raise ValueError('invalid uniform distribution parameters; upper should be greater than lower.')
        if not( (self.endpoints.lower() == 'none') or (self.endpoints.lower() == 'lower') or (self.endpoints.lower() == 'upper') ):
            raise ValueError('invalid selection for endpoints') 
        self.parent = uniform(loc=(self.lower), scale=(self.upper - self.lower))
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, lower=self.lower, upper=self.upper, mean=self.mean, variance=self.variance, skewness=self.skewness, kurtosis=self.kurtosis, x_range_for_pdf=self.x_range_for_pdf, order=self.order, endpoints=self.endpoints)
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
        ab =  jacobi_recurrence_coefficients(0., 0., self.lower, self.upper, order)
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
