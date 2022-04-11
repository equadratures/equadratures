"""The Beta distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import beta
RECURRENCE_PDF_SAMPLES = 8000

class Beta(Distribution):
    """
    The class defines a Beta object. It is the child of Distribution.

    :param 
    """
    def __init__(self, **kwargs):
        first_arg = ['alpha', 'shape_parameter_A', 'shape_A']
        second_arg = ['beta', 'shape_parameter_B', 'shape_B']
        third_arg = ['lower', 'low', 'bottom']
        fourth_arg = ['upper','up', 'top']  
        fifth_arg = ['order', 'orders', 'degree', 'degrees']
        sixth_arg = ['endpoints', 'endpoint']
        seventh_arg = ['variable']
        self.name = 'beta'
        self.lower = None
        self.upper = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.shape_A = value 
            if second_arg.__contains__(key):
                self.shape_B = value 
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

        if self.lower is None and self.upper is None:
            self.lower = 0. # Standard beta distribution defn'
            self.upper = 1.
        if self.lower is None or self.upper is None:
            raise ValueError('lower or upper bounds have not been specified!')
        if self.upper <= self.lower:
            raise ValueError('invalid beta distribution parameters: upper should be greater than lower.')
        if self.shape_A <= 0 or self.shape_B <= 0:
            raise ValueError('invalid beta distribution parameters: shape parameters must be positive!')
        loc = self.lower
        scale = self.upper - self.lower
        self.parent = beta(self.shape_A, self.shape_B, loc=loc, scale=scale)
        self.mean, self.variance, self.skewness, self.kurtosis = beta.stats(self.shape_A, self.shape_B, loc=loc, scale=scale, moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
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
        A description of the beta distribution.

        :param Beta self:
            An instance of the beta class.
        :return:
            A string describing the beta distribution.
        """
        text = ("is a beta distribution is defined over a support; given here as " + str(self.lower) + ", to " + str(self.upper) + \
                ". It has two shape parameters, given here to be " + str(self.shape_A) + " and " + str(self.shape_B) + ".")
        return text
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the beta distribution.

        :param Beta self:
            An instance of the Beya class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the beta distribution.
        """
        ab = jacobi_recurrence_coefficients(self.shape_B - 1.0
                                            , self.shape_A - 1.0
                                            , self.lower, self.upper, order)
        return ab
