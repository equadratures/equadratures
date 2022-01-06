"""The Rayleigh distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import rayleigh
RECURRENCE_PDF_SAMPLES = 8000
class Rayleigh(Distribution):
    """
    The class defines a Rayleigh object. It is the child of Distribution.

    :param double scale:
		Scale parameter of the Rayleigh distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['scale', 'sigma', 'shape_parameter_A']
        second_arg = ['order', 'orders', 'degree', 'degrees']
        third_arg = ['endpoints', 'endpoint']
        fourth_arg = ['variable']
        self.name = 'rayleigh'
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.scale = value
            if second_arg.__contains__(key):
                self.order = value
            if third_arg.__contains__(key):
                self.endpoints = value
            if fourth_arg.__contains__(key):
                self.variable = value

        if self.scale is None:
            self.scale = 1.0

        self.bounds = np.array([0.0, np.inf])
        if self.scale < 0:
            raise ValueError('Invalid parameters in Rayleigh distribution. Scale should be positive.')
        self.parent = rayleigh(scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0.0, 8.0 * self.scale, RECURRENCE_PDF_SAMPLES)
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
        A description of the Rayleigh distribution.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :return:
            A string describing the Rayleigh distribution.
        """
        text = ("is a Rayleigh distribution; characterised by its scale parameter, which has been set to " + \
                str(self.scale) + ".")
        return text
