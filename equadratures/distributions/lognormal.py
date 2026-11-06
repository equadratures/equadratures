"""The Lognormal distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import lognorm
RECURRENCE_PDF_SAMPLES = 50000
class Lognormal(Distribution):
    """
    The class defines a Lognormal object. It is the child of Distribution.
    See "scipy.stats.lognorml" usage.

    :param double mean:
		Mean of the normal distribution.
	:param double standard_deviation:
		Standard deviation of the normal distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['mean', 'mu', 'shape_parameter_A']
        second_arg = ['standard_deviation', 'std', 'sigma', 'shape_parameter_B']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'lognormal'
        self.mu = None
        self.sigma = None
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.mu = value
            if second_arg.__contains__(key):
                self.sigma = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.mu is None or self.sigma is None:
            raise ValueError('mean or standard deviation have not been specified!')
        if self.sigma <= 0:
            raise ValueError('Invalid parameters in lognormal distribution. Standard deviation should be positive.')

        self.bounds = np.array([0.0, np.inf])

        self.parent = lognorm(scale=np.exp(self.mu), s=self.sigma)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(-15.0 * self.sigma, 15.0*self.sigma, RECURRENCE_PDF_SAMPLES) + self.mean
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
        A description of the Lognormal distribution.

        :param Lognormal self:
            An instance of the Lognormal class.
        :return:
            A string describing the Lognormal distribution.
        """
        text = ("is a Lognormal distribution is characterised by its mean parameter, which here is " + \
                str(self.mu) + " and its standard deviation, given by " + str(self.sigma) + ".")
        return text
