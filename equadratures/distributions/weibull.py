"""The Weibull distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import weibull_min
RECURRENCE_PDF_SAMPLES = 8000

class Weibull(Distribution):
    """
    The class defines a Weibull object. It is the child of Distribution.

    :param double scale:
		Upper bound of the support of the Weibull distribution.
    :param double shape:
		Lower bound of the support of the Weibull distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['scale', 'lamda', 'shape_parameter_A']
        second_arg = ['shape', 'k', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'weibull'
        self.scale = None
        self.shape = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.scale = value
            if second_arg.__contains__(key):
                self.shape = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.scale is None:
            self.scale = 1.0
        if self.shape is None:
            self.shape = 1.0

        self.bounds = np.array([0.0, np.inf])
        if self.scale < 0 or self.shape < 0:
            raise ValueError('Invalid parameters in Weibull distribution. Shape and Scale should be positive.')
        self.parent = weibull_min(c=self.shape, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0, self.scale*10, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.bounds[0], \
                        upper=self.bounds[1], \
                        scale=self.scale, \
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
        A description of the Weibull distribution.

        :param Weibull self:
            An instance of the Weibull class.
        :return:
            A string describing the Weibull distribution.
        """
        text = ("is a Weibull distribution with a scale parameter of " + str(self.scale) + \
                " and a shape parameter of " + str(self.shape) + ".")
        return text
