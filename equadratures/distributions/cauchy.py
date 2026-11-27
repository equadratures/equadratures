"""The Cauchy distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import cauchy
RECURRENCE_PDF_SAMPLES = 8000

class Cauchy(Distribution):
    """
    The class defines a Cauchy object. It is the child of Distribution.

    :param double location:
		Location parameter of the Cauchy distribution.
    :param double scale:
		Scale parameter of the Cauchy distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['location', 'loc', 'shape_parameter_A']
        second_arg = ['scale', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'cauchy'
        self.loc = None
        self.scale = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.loc = value
            if second_arg.__contains__(key):
                self.scale = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.loc is None or self.scale is None:
            raise ValueError('location or scale have not been specified!')
        if self.scale <= 0:
            raise ValueError('invalid Cauchy distribution parameters; scale should be positive.')

        if self.scale is None:
            self.scale = 1.0

        self.lower = -np.inf
        self.upper =  np.inf
        self.bounds = np.array([self.lower, self.upper])

        self.x_range_for_pdf = np.linspace(-15*self.scale, 15*self.scale, RECURRENCE_PDF_SAMPLES)
        self.parent = cauchy(loc=self.loc, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        super().__init__(name=self.name, \
                        lower=self.lower, \
                        upper=self.upper, \
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
        A description of the Cauchy distribution.

        :param Cauchy self:
            An instance of the Cauchy class.
        :return:
            A string describing the Cauchy distribution.
        """
        text = ("is a Cauchy distribution that by definition has an undefined mean and variance; its " \
                "location parameter is " + str(self.loc) + ", and its scale parameter is " + str(self.scale) + ".")
        return text
