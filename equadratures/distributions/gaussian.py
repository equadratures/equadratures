"""The Gaussian / Normal distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import norm
RECURRENCE_PDF_SAMPLES = 8000
class Gaussian(Distribution):
    """
    The class defines a Gaussian object. It is the child of Distribution.

    :param double shape_parameter_A:
		Mean of the Gaussian distribution.
	:param double shape_parameter_B:
		Variance of the Gaussian distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['mean', 'mu', 'shape_parameter_A']
        second_arg = ['variance', 'var', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'gaussian'
        self.mean = None 
        self.variance = None 
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.mean = value
            if second_arg.__contains__(key):
                self.variance = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.mean is None or self.variance is None:
            raise ValueError('mean or variance have not been specified!')
        if self.variance <= 0:
            raise ValueError('invalid Gaussian distribution parameters; variance should be positive.')
        self.sigma = np.sqrt(self.variance)
        self.x_range_for_pdf = np.linspace(-15.0 * self.sigma, 15.0*self.sigma, RECURRENCE_PDF_SAMPLES) + self.mean
        self.parent = norm(loc=self.mean, scale=self.sigma)
        _, _ , self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        super().__init__(name=self.name, \
                        lower=-np.inf, \
                        upper=np.inf, \
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
        text = ("is a Gaussian distribution with a mean of " + str(self.mean) + " and a variance of " + str(self.variance) \
                + ".")
        return text

