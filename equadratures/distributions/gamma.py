"""The Gamma distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import gamma
RECURRENCE_PDF_SAMPLES = 8000
class Gamma(Distribution):
    """
    The class defines a Gamma object. It is the child of Distribution.

    :param double shape:
		Shape parameter of the gamma distribution.
    :param double scale:
		Scale parameter of the gamma distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['shape', 'k', 'shape_parameter_A']
        second_arg = ['scale', 'theta', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'gamma'
        self.shape = None
        self.scale = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.shape = value
            if second_arg.__contains__(key):
                self.scale = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.shape is None:
            self.shape = 1.0
        if self.scale is None:
            self.scale = 1.0

        self.bounds = np.array([0.0, np.inf])
        if self.shape < 0 or self.scale < 0:
            raise ValueError('Invalid parameters in Gamma distribution. Shape and Scale should be positive.')
        self.parent = gamma(a=self.shape, scale=self.scale)
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
        A description of the gamma distribution.

        :param Gamma self:
            An instance of the Gamma class.
        :return:
            A string describing the gamma distribution.
        """
        text = ("is a gamma distribution with a shape parameter of " + str(self.shape) + ", and a scale " \
                "parameter of " + str(self.scale) + ".")
        return text
