"""The Exponential distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
from scipy.stats import expon
RECURRENCE_PDF_SAMPLES = 8000

class Exponential(Distribution):
    """
    The class defines a Exponential object. It is the child of Distribution.

    :param double rate:
		Rate parameter of the Exponential distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, rate=None, data=None):
        if rate is None:
            if data is None:
                self.rate = 1.0
                self.data = None
            else:
                self.rate = None
                self.data = data
        else:
            self.rate = rate
            self.data = data

        if self.data is not None:
            params=expon.fit(data)
            self.rate=1.0 / params[1]
        if (self.rate is not None) and (self.rate > 0.0):
            self.parent = expon(scale=1.0 / self.rate, loc=0)
            self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
            self.bounds = np.array([0.0, np.inf])
            self.x_range_for_pdf = np.linspace(0.0, 20*self.rate, RECURRENCE_PDF_SAMPLES)
        else:
            raise ValueError('Invalid parameters in exponential distribution. Rate should be positive.')

    def get_description(self):
        """
        A description of the Exponential distribution.

        :param Exponential self:
            An instance of the Exponential class.
        :return:
            A string describing the Exponential distribution.
        """
        text = "is an exponential distribution with a rate parameter of"+str(self.rate)+"."
        return text
    def get_pdf(self, points=None):
        """
        An exponential probability density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param matrix points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Probability density values along the support of the exponential distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getPDF method')
    def get_icdf(self, xx):
        """
        An inverse exponential cumulative density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the exponential distribution.
        """
        return self.parent.ppf(xx)
    def get_cdf(self, points=None):
        """
        An exponential cumulative density function.

        :param Exponential self:
            An instance of the Exponential class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Cumulative density values along the support of the exponential distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_samples(self, m=None):
        """
        Generates samples from the Exponential distribution.

         :param Expon self:
             An instance of the Exponential class.
         :param integer m:
              Number of random samples. If no value is provided, a default of 5e05 is assumed
         :return:
             A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)
