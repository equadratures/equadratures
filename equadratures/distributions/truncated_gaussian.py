"""The Truncated Gaussian distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import truncnorm
RECURRENCE_PDF_SAMPLES = 8000

class TruncatedGaussian(Distribution):
    """
    The class defines a Truncated-Gaussian object. It is the child of Distribution.
    :param double mean:
		Mean of the truncated Gaussian distribution.
	:param double variance:
		Variance of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the truncated Gaussian distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['mean', 'mu', 'shape_parameter_A']
        second_arg = ['variance', 'var', 'shape_parameter_B']
        third_arg = ['lower', 'low']
        fourth_arg = ['upper', 'up']
        fifth_arg = ['order', 'orders', 'degree', 'degrees']
        sixth_arg = ['endpoints', 'endpoint']
        seventh_arg = ['variable']
        self.name = 'truncated-gaussian'
        self.mean = None
        self.variance = None
        self.lower = None
        self.upper = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.mean = value
            if second_arg.__contains__(key):
                self.variance = value
            if third_arg.__contains__(key):
                self.lower = value
            if fourth_arg.__contains__(key):
                self.upper = value
            if fifth_arg.__contains__(key):
                self.order = value
            if sixth_arg.__contains__(key):
                self.endpoints = value
            if seventh_arg.__contains__(key):
                self.variable = value

        if self.mean is None:
            self.mean = 0.0
        if self.variance is None:
            self.variance = 1.0
        if self.lower is None:
            self.lower = -3.0
        if self.upper is None:
            self.upper = 3.0

        self.std = np.sqrt(self.variance)
        a = (self.lower - self.mean) / self.std
        b = (self.upper - self.mean) / self.std
        self.parent = truncnorm(a, b, loc=self.mean, scale=self.std)
        self.bounds = np.array([self.lower, self.upper])
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        super().__init__(name=self.name, \
                        lower=self.lower, \
                        upper=self.upper, \
                        mean=self.mean, \
                        variance=self.variance, \
                        skewness=self.skewness, \
                        kurtosis=self.kurtosis, \
                        x_range_for_pdf=self.x_range_for_pdf, \
                        order=self.order, \
                        endpoints=self.endpoints, \
                        variable=self.variable, \
                        scipyparent=self.parent)
    def get_description(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = ("a truncated Gaussian distribution with a center of " + str(self.mean) + " and a scale of " \
                + str(self.std) + ", and a lower bound of " + str(self.lower) + " and an upper bound of " + \
                str(self.upper) + ".")
        return text
