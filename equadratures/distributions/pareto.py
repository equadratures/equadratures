"""The Pareto distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import pareto
RECURRENCE_PDF_SAMPLES = 8000
class Pareto(Distribution):
    """
    The class defines a Rayleigh object. It is the child of Distribution.

    :param double scale:
		Scale parameter of the Rayleigh distribution.
    """
    def __init__(self, scale):
        self.scale = scale
        self.bounds = np.array([, np.inf])
        if self.scale is not None:
            if self.scale > 0:
                self.mean = self.scale * np.sqrt(np.pi / 2.0)
                self.variance = self.scale**2 * (4.0 - np.pi)/ 2.0
                self.skewness = 2.0 * np.sqrt(np.pi) * (np.pi - 3.0) / ((4.0 - np.pi)**(1.5))
                self.kurtosis = -(6 * np.pi**2 - 24 * np.pi + 16.0 )/( (4 - np.pi)**(1.5)) + 3.0
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
        return pareto.ppf(xx, loc=0, scale=self.scale)

    def get_description(self):
        """
        A description of the Rayleigh distribution.

        :param Rayleigh self:
            An instance of the Rayleigh class.
        :return:
            A string describing the Rayleigh distribution.
        """
        text = "A Rayleigh distribution is characterised by its scale parameter, which is"+str(self.scale)+"."
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
        return pareto.pdf(points, loc=0, scale=self.scale )


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
        return pareto.cdf(points, loc=0, scale=self.scale )

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
        return pareto.rvs(loc=0.0, scale=self.scale, size=number, random_state=None)
