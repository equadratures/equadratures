"""The Chebyshev / Arcsine distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import arcsine
RECURRENCE_PDF_SAMPLES = 8000
class Chebyshev(Distribution):
    """
    The class defines a Chebyshev object. It is the child of Distribution.

    :param double lower:
		Lower bound of the support of the Chebyshev (arcsine) distribution.
	:param double upper:
		Upper bound of the support of the Chebyshev (arcsine) distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['lower', 'low']
        second_arg = ['upper', 'up']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'chebyshev'
        self.location = None
        self.scale = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.lower = value
            if second_arg.__contains__(key):
                self.upper = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.lower is None:
            self.lower = 0.0
        if self.upper is None:
            self.upper = 1.0

        print(self.lower, self.upper)

        if self.lower > self.upper:
            raise ValueError('Invalid Chebyshev distribution parameters. Lower should be smaller than upper.')

        self.bounds = np.array([self.lower, self.upper])
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        loc = self.lower
        scale = self.upper - self.lower

        self.parent = arcsine(loc=loc, scale=scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.shape_parameter_A = -0.5
        self.shape_parameter_B = -0.5
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
        A description of the Chebyshev (arcsine) distribution.

        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :return:
            A string describing the Chebyshev (arcsine) distribution.
        """
        text = ("is a Chebyshev or arcsine distribution that is characterised by its lower bound, which " \
                "is " + str(self.lower) + " and its upper bound, which is " + str(self.upper) + ".")
        return text
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the Chebyshev distribution.
        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the Chebyshev distribution.
        """
        ab = jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper, order)
        return ab