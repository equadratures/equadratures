"""The Cauchy distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
from scipy.stats import cauchy
RECURRENCE_PDF_SAMPLES = 8000

class Cauchy(Distribution):
    """
    The class defines a Cauchy object. It is the child of Distribution.

    :param double location:
		Location parameter of the Cauchy distribution.
    :param double scale:
		Scale parameter of the Cauchy distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, location=None, scale=None, data=None):

        if data is None:
            self.location = location
            self.data = None
        else:
            self.location = None
            self.data = data

        if scale is None:
            if data is None:
                self.scale = 1.0
                self.data = None
            else:
                self.scale = None
                self.data =data
        else:
            self.scale = scale
            self.data = None

        if self.data is not None:
            params=cauchy.fit(self.data)
            self.location=params[0]
            self.scale=params[1]

        self.bounds = np.array([-np.inf, np.inf])
        self.mean = np.nan
        self.variance = np.nan
        self.skewness = np.nan
        self.kurtosis = np.nan

        self.x_range_for_pdf = np.linspace(-15*self.scale, 15*self.scale, RECURRENCE_PDF_SAMPLES)
        self.parent = cauchy(loc=self.location, scale=self.scale)
        # self.mean = np.mean(self.get_samples(m=1000))
        # self.variance = np.var(self.get_samples(m=1000))

    def get_description(self):
        """
        A description of the Cauchy distribution.

        :param Cauchy self:
            An instance of the Cauchy class.
        :return:
            A string describing the Cauchy distribution.
        """
        text = "is a Cauchy distribution that by definition has an undefined mean and variance; its location parameter is "+str(self.location)+", and its scale parameter is "+str(self.scale)+"."
        return text

    def get_pdf(self, points=None):
        """
        A Cauchy probability density function.

        :param Cauchy self:
            An instance of the Cauchy class.
        :param array points:
            Array of points for defining the probability density function.
        :return:
            An array of N values over the support of the distribution.
        :return:
            Probability density values along the support of the Cauchy distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getPDF method')
    def get_cdf(self, points=None):
        """
        A Cauchy cumulative density function.

        :param Cauchy self:
            An instance of the Cauchy class.
        :param array points:
            Array of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the Cauchy distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_icdf(self, xx):
        """
        An inverse Cauchy cumulative density function.

        :param Cauchy self:
            An instance of the Cauchy class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0, 1].
        :return:
            Inverse CDF samples associated with the Cauchy distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m):
        """
         Generates samples from the Gaussian distribution.
        :param Gaussian self:
            An instance of the Gaussian class.
        :param integer m:
            Number of random samples. If no value is provided, a default of     5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)
