"""The Pareto distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import pareto
RECURRENCE_PDF_SAMPLES = 50000
class Pareto(Distribution):
    """
    The class defines a Pareto object. It is the child of Distribution.

    :param double shape:
		The shape parameter associated with the Pareto distribution.
    :param double scale:
		The scale parameter associated with the Pareto distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['shape', 'alpha', 'shape_parameter_A']
        second_arg = ['scale', 'xm', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'pareto'
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

        self.bounds = np.array([self.scale, np.inf])
        if self.shape < 0 or self.scale < 0:
            raise ValueError('Invalid parameters in Pareto distribution. Shape and Scale should be positive.')
        self.parent = pareto(b=self.shape, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.scale, self.shape + 20.0, RECURRENCE_PDF_SAMPLES)
        self.Scale = self.scale
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
        A description of the Pareto distribution.

        :param Pareto self:
            An instance of the Pareto class.
        :return:
            A string describing the Pareto distribution.
        """
        text = ("is a Pareto distribution is characterised by its shape parameter, which here is " + \
                str(self.shape) + " and its scale, given by " + str(self.Scale) + ".")
        return text
