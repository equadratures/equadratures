"""The Logistic distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import logistic
RECURRENCE_PDF_SAMPLES = 50000
class Logistic(Distribution):
    """
    The class defines a Logistic object. It is the child of Distribution.

    :param int shape_parameter:
		The shape parameter associated with the Logistic distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['location', 'loc', 'mu', 'shape_parameter_A']
        second_arg = ['scale', 's', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'logistic'
        self.location = None
        self.scale = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.location = value
            if second_arg.__contains__(key):
                self.scale = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.location is None:
            self.location = 0.0
        if self.scale is None:
            self.scale = 1.0

        self.bounds = np.array([-np.inf, np.inf])
        if self.scale < 0:
            raise ValueError('Invalid parameter in Logistic distribution. Scale should be positive.')

        self.parent = logistic(loc=self.location, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.location - 10.0, 20.0 + self.location, RECURRENCE_PDF_SAMPLES)
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
        A description of the Logistic distribution.

        :param Logistic self:
            An instance of the Logistic class.
        :return:
            A string describing the Logistic distribution.
        """
        text = ("is a Logistic distribution is characterised by its location parameter, which here is " + \
                str(self.location) + " and its scale, given by " + str(self.scale) + ".")
        return text
