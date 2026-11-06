""" The Analytical distribution"""
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
from equadratures.distributions.template import Distribution
import numpy as np
import scipy.stats as stats
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
RECURRENCE_PDF_SAMPLES = 8000

class Data(Distribution):
    """ The class defines a Analytical object.

        :param Weight weight_function:
              An instance of the Weight class.
    """
    def __init__(self, **kwargs):
        first_arg = ['data']
        second_arg = ['lower']
        third_arg = ['upper']
        fourth_arg = ['order', 'orders', 'degree', 'degrees']
        self.name = 'data'
        self.mean = None 
        self.variance = None 
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.data = value
            if second_arg.__contains__(key):
                self.lower = value
            if third_arg.__contains__(key):
                self.upper = value
            if fourth_arg.__contains__(key):
                self.order = value
        self.parent = stats.gaussian_kde(dataset=self.data, bw_method='scott') 
        #self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.lower, \
                        upper=self.upper, \
                        mean=self.mean, \
                        variance=self.variance, \
                        skewness=0., \
                        kurtosis=0., \
                        x_range_for_pdf=self.x_range_for_pdf, \
                        order=self.order, \
                        endpoints=self.endpoints, \
                        variable=self.variable, \
                        scipyparent=self.parent)
    def get_description(self):
        """ A destription of Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :return:
                A string describing the Analytical distribution.
        """
        text = "is a Analytical distribution defined over a support from "+str(self.lower)+" to "+str(self.upper)+". \
            It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
        return text
