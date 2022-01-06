"""The Exponential distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
from scipy.stats import expon
RECURRENCE_PDF_SAMPLES = 8000

class Exponential(Distribution):
    """
    The class defines a Exponential object. It is the child of Distribution.

    :param double rate:
		Rate parameter of the Exponential distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['rate', 'lamda', 'shape_parameter_A']
        second_arg = ['order', 'orders', 'degree', 'degrees']
        third_arg = ['endpoints', 'endpoint']
        fourth_arg = ['variable']
        self.name = 'exponential'
        self.rate = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.rate = value
                self.shape_parameter_A = value
            if second_arg.__contains__(key):
                self.order = value
            if third_arg.__contains__(key):
                self.endpoints = value
            if fourth_arg.__contains__(key):
                self.variable = value

        if self.rate is None:
            self.rate = 1.0

        if (self.rate is not None) and (self.rate > 0.0):
            self.parent = expon(scale=1.0 / self.rate)
            self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
            self.bounds = np.array([0.0, np.inf])
            self.x_range_for_pdf = np.linspace(0.0, 20*self.rate, RECURRENCE_PDF_SAMPLES)
            super().__init__(name=self.name, \
                            lower=self.bounds[0], \
                            upper=self.bounds[1], \
                            rate=self.rate, \
                            mean=self.mean, \
                            variance=self.variance, \
                            skewness=self.skewness, \
                            kurtosis=self.kurtosis, \
                            x_range_for_pdf=self.x_range_for_pdf, \
                            order=self.order, \
                            endpoints=self.endpoints, \
                            variable=self.variable, \
                            scipyparent=self.parent)
        else:
            raise ValueError('Invalid parameters in exponential distribution. Rate should be positive.')
    def get_description(self):
        """
        A description of the Exponential distribution.

        :param Exponential self:
            An instance of the Exponential class.
        :return:
            A string describing the Exponential distribution.
        """
        text = "is an exponential distribution with a rate parameter of " + str(self.rate) + "."
        return text
