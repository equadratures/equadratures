"""The Gaussian / Normal distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import norm
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

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
        self.name = 'gaussian'
        self.mean = None 
        self.variance = None 
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.mean = value
            if second_arg.__contains__(key):
                self.variance = value
        if self.mean is None or self.variance is None:
            raise ValueError('mean or variance have not been specified!')
        if self.variance <= 0:
            raise ValueError('invalid Gaussian distribution parameters; variance should be positive.')
        self.sigma = np.sqrt(self.variance)
        self.x_range_for_pdf = np.linspace(-15.0 * self.sigma, 15.0*self.sigma, RECURRENCE_PDF_SAMPLES) + self.mean
        self.parent = norm(loc=self.mean, scale=self.sigma)
        super().__init__(name=self.name, mean=self.mean, variance=self.variance, x_range_for_pdf=self.x_range_for_pdf)
    def get_description(self):
        """
        A description of the Gaussian.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            A string describing the Gaussian.
        """
        text = "is a Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+"."
        return text
    def get_samples(self, m=None):
        """
        Generates samples from the Gaussian distribution.
        :param Gaussian self:
            An instance of the Gaussian class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)
    def get_pdf(self, points=None):
        """
        A Gaussian probability distribution.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gaussian distribution.
        """
        return self.parent.pdf(points)
    def get_cdf(self, points=None):
        """
        A Gaussian cumulative density function.

	    :param Gaussian self:
            An instance of the Gaussian class.
        :param array points
            Points for which the cumulative density function is required.
        :return:
            Gaussian cumulative density values.
        """
        return self.parent.cdf(points)
    def get_icdf(self, xx):
        """
        An inverse Gaussian cumulative density function.

        :param Gaussian self:
            An instance of the Gaussian class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the Gaussian distribution.
        """
        return self.parent.ppf(xx)
