"""The Rayleigh distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import rayleigh
RECURRENCE_PDF_SAMPLES = 8000
class Rayleigh(Distribution):
    """
    The class defines a Rayleigh object. It is the child of Distribution.

    :param double scale:
		Scale parameter of the Rayleigh distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, scale, data):
        if scale is None:
            if data is None:
                self.scale = 1.0
                self.data = None
            else:
                self.data = data
                self.scale = None
        else:
            self.scale = scale
            self.data = None

        self.bounds = np.array([0.999, np.inf])
        if self.data is not None:
            params=rayleigh.fit(data)
            self.scale=params[1]
        if self.scale < 0:
            raise ValueError('Invalid parameters in Rayleigh distribution. Scale should be positive.')
        self.parent = rayleigh(scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0.0, 8.0 * self.scale, RECURRENCE_PDF_SAMPLES)

    def get_icdf(self, xx):
        """
        A Rayleigh inverse cumulative density function.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :param array xx:
            Points at which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Rayleigh distribution.
        """
        return self.parent.ppf(xx)

    def get_description(self):
        """
        A description of the Rayleigh distribution.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :return:
            A string describing the Rayleigh distribution.
        """
        text = "is a Rayleigh distribution; characterised by its scale parameter, which has been set to "+str(self.scale)+"."
        return text

    def get_pdf(self, points=None):
        """
        A Rayleigh probability density function.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :param array points:
            Points at which the PDF needs to be evaluated.
        :return:
            Probability density values along the support of the Rayleigh distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError('Please digit an input for get_pdf method')


    def get_cdf(self, points=None):
        """
        A Rayleigh cumulative density function.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :param array points:
            Points at which the CDF needs to be evaluated.
        :return:
            Cumulative density values along the support of the Rayleigh distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError('Please digit an input for get_cdf method')

    def get_samples(self, m=None):
        """
         Generates samples from the Rayleigh distribution.

         :param rayleigh self:
             An instance of the Rayleigh class.
         :param integer m:
             Number of random samples. If no value is provided, a default of     5e5 is assumed.
         :return:
             A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)
