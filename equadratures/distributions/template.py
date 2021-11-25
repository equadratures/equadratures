"""The Distribution template."""
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
from equadratures.parentparameter import ParentParameter
import numpy as np

PDF_SAMPLES = 500000

class Distribution(ParentParameter):
    """
    The class defines a Distribution object. It serves as a template for all distributions.


    """
    def __init__(self, name, x_range_for_pdf, mean=None, variance=None, skewness=None, kurtosis=None, lower=-np.inf, upper=np.inf, rate=None, scale=None, order=2, variable='parameter'):
        self.name = name
        self.mean = mean 
        self.variance = variance 
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.lower = lower 
        self.upper = upper 
        self.x_range_for_pdf = x_range_for_pdf
        self.rate = rate 
        self.scale = scale
        self.bounds = [self.lower, self.upper]
        self.order = order
        self.variable = variable
        super().__init__(distribution=name, order=order, variable=variable, endpoints=None)
    def __eq__(self, second_distribution):
        """
        Returns a boolean to clarify if two distributions are the same.

        :param Distribution self:
                An instance of the Distribution class.
        :param Distribution second_distribution:
                A second instance of the Distribution class.
        """
        if self.name == second_distribution.name and \
            self.mean == second_distribution.mean and \
            self.variance == second_distribution.variance and \
            self.lower == second_distribution.lower and \
            self.upper == second_distribution.upper and \
            self.rate == self.rate and \
            self.scale == self.scale and \
            self.x_range_for_pdf == self.x_range_for_pdf:
            return True 
        else:
            False
    def get_description(self):
        """
        Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_pdf(self, points=None):
        """
        Returns the PDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_cdf(self, points=None):
        """
        Returns the CDF of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_icdf(self, xx):
        """
        An inverse cumulative density function.

        :param Distribution self:
                An instance of the distribution class.
        :param xx:
                A numpy array of uniformly distributed samples between [0,1].
        :return:
                Inverse CDF samples associated with the gamma distribution.
        """
        pass
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the distribution

        :param Distribution self:
            An instance of the distribution class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the distribution.
        """
        w_pdf = self.get_pdf(self.x_range_for_pdf)
        ab = custom_recurrence_coefficients(self.x_range_for_pdf, w_pdf, order)
        return ab
    def get_samples(self, m=None):
        """
        Generates samples from the distribution.

        :param Distribution self:
            An instance of the distribution class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is None:
            number_of_random_samples = PDF_SAMPLES
        else:
            number_of_random_samples = m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.get_icdf(uniform_samples)
        return yy
