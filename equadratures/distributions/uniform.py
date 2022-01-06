""" Please add a file description here"""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import uniform
RECURRENCE_PDF_SAMPLES = 8000

class Uniform(Distribution):
    """
    The class defines a Uniform object. It is the child of Distribution.

    :param double mean:
		Mean of the Gaussian distribution.
	:param double variance:
		Variance of the Gaussian distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['lower', 'low', 'bottom']
        second_arg = ['upper','up', 'top']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'uniform'
        self.lower = None 
        self.upper = None 
        self.order = 1
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

        if self.lower is None or self.upper is None:
            raise ValueError('lower or upper bounds have not been specified!')
        if self.upper <= self.lower:
            raise ValueError('invalid uniform distribution parameters; upper should be greater than lower.')
        if not( (self.endpoints.lower() == 'none') or (self.endpoints.lower() == 'lower') or (self.endpoints.lower() == 'upper')  \
            or (self.endpoints.lower() == 'both') ):
            raise ValueError('invalid selection for endpoints') 
        self.parent = uniform(loc=(self.lower), scale=(self.upper - self.lower))
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
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
        A description of the Gaussian.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            A string describing the Gaussian.
        """
        text = "is a uniform distribution over the support "+str(self.lower)+" to "+str(self.upper)+"."
        return text
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the uniform distribution.
        :param Uniform self:
            An instance of the Uniform class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the uniform distribution.
        """
        ab =  jacobi_recurrence_coefficients(0., 0., self.lower, self.upper, order)
        return ab
