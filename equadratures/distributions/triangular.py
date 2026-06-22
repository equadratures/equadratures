"""The Triangular distrubution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import triang
RECURRENCE_PDF_SAMPLES = 8000

class Triangular(Distribution):
    """
    The class defines a Triangular object.

    :param double lower:
        Lower bound of the support of the distribution.
    :param double upper:
        Upper bound of the support of the distribution.
    :param double mode:
        Mode of the distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['lower', 'a']
        second_arg = ['upper', 'b']
        third_arg = ['mode', 'c', 'shape_parameter_A']
        fourth_arg = ['order', 'orders', 'degree', 'degrees']
        fifth_arg = ['endpoints', 'endpoint']
        sixth_arg = ['variable']
        self.name = 'triangular'
        self.lower = None
        self.upper = None
        self.mode = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.lower = value
            if second_arg.__contains__(key):
                self.upper = value
            if third_arg.__contains__(key):
                self.mode = value
            if fourth_arg.__contains__(key):
                self.order = value
            if fifth_arg.__contains__(key):
                self.endpoints = value
            if sixth_arg.__contains__(key):
                self.variable = value

        if (self.lower is None) or (self.upper is None) or (self.mode is None):
            raise ValueError('Invalid parameters in Triangular distribution. Missing either Lower, Mode, or Upper parameters.')

        if self.lower > self.upper:
            raise ValueError('Invalid parameters in Triangular distribution. Lower should be smaller than Upper.')
        if not (self.lower <= self.mode <= self.upper):
            raise ValueError('Invalid parameters in Triangular distribution. Mode must be between Lower and Upper.')

        self.bounds = np.array([self.lower, self.upper])
        self.scale = self.upper - self.lower # scale
        self.shape = (self.mode - self.lower) / (self.upper - self.lower) # c

        self.parent = triang(loc=self.lower, scale=self.scale, c=self.shape)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.bounds[0], \
                        upper=self.bounds[1], \
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
        Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        text = ("is a triangular distribution with a mode of " + str(self.mode) + " over the support " + \
                str(self.lower) + " to " + str(self.upper) + ".")
        return text
