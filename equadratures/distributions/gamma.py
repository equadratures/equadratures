"""The Gamma distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import gamma
RECURRENCE_PDF_SAMPLES = 8000
class Gamma(Distribution):
    """
    The class defines a Gamma object. It is the child of Distribution.

    :param double shape:
		Shape parameter of the gamma distribution.
    :param double scale:
		Scale parameter of the gamma distribution.
    """
    def __init__(self, shape=None, scale=None, data=None):
        if shape is None:
            if data is None:
                self.shape = 1.0
                self.data = None
            else:
                self.shape = None
                self.data = data
        else:
            self.shape = shape
            self.data = None

        if scale is None:
            if data is None:
                self.scale = 1.0
                self.data = None
            else:
                self.scale = None
                self.data = data
        else:
            self.scale = scale
            self.data = None

        if self.data is not None:
            params = gamma.fit(data)
            self.shape = params[0]
            self.scale = params[2]

        self.bounds = np.array([0.0, np.inf])
        if self.shape < 0 or self.scale < 0:
            raise ValueError('Invalid parameters in Gamma distribution. Shape and Scale should be positive.')
        self.parent = gamma(a=self.shape, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0, self.scale*10, RECURRENCE_PDF_SAMPLES)

    def get_description(self):
        """
        A description of the gamma distribution.

        :param Gamma self:
            An instance of the Gamma class.
        :return:
            A string describing the gamma distribution.
        """
        text = "is a gamma distribution with a shape parameter of "+str(self.shape)+", and a scale parameter of "+str(self.scale)+"."
        return text

    def get_pdf(self, points=None):
        """
        A gamma probability density function.

        :param Gamma self:
            An instance of the Gamma class.
        :param matrix points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the Gamma distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getPDF method')

    def get_cdf(self, points=None):
        """
        A gamma cumulative density function.

        :param Gamma self:
            An instance of the Gamma class.
        :param matrix points:
            Matrix of points for defining the gamma cumulative density function.
        :return:
            An array of N equidistant values over the support of the gamma distribution.
        :return:
            Cumulative density values along the support of the gamma distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_icdf(self, xx):
        """
        A gamma inverse cumulative density function.

        :param gamma self:
            An instance of Gamma class.
        :param xx:
            An array of points at which the inverse of cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Gamma distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """
         Generates samples from the Gamma distribution.

         :param Gamma self:
             An instance of the Gamma class.
         :param integer m:
             Number of random samples. If no value is provided, a default of     5e5 is assumed.
         :return:
             A N-by-1 vector that contains the samples.
        """
        if m is not None:
           number = m
        else:
           number = 500000
        return self.parent.rvs(size = number)
