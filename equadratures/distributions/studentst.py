"""The Student's T distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.special import erf, erfinv, gamma, gammainc
from scipy.stats import t
RECURRENCE_PDF_SAMPLES = 50000
class Studentst(Distribution):
    """
    The class defines a Studentst object. It is the child of Distribution.

    :param int dofs:
		Degrees of freedom for the Student's T distribution.
    """
    def __init__(self, dofs):
        self.dofs = dofs
        if self.dofs is not None:
            if self.dofs > 0:
                self.bounds = np.array([-np.inf, np.inf])
                mean, var, skew, kurt = t.stats(df=self.dofs, moments='mvsk')
                self.parent = t(df=self.dofs)
                self.mean = mean
                self.variance = var
                self.skewness = skew
                self.kurtosis = kurt
                self.x_range_for_pdf = np.linspace(-5.0, 5.0,RECURRENCE_PDF_SAMPLES)
                self.parent = t(self.dofs)
    def get_description(self):
        """
        A description of the Studentst distribution.

        :param Studentst self:
            An instance of the Student's T class.
        :return:
            A string describing the Student's T distribution.
        """
        text = "is a student's t distribution; characterised by its degrees of freedom, which here is"+str(self.dofs)+"."
        return text
    def get_pdf(self, points=None):
        """
        A Studentst probability density function.

        :param Studentst self:
            An instance of the Student's T class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Student's T distribution.
        :return:
            Probability density values along the support of the Student's T distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getPDF method')
    def get_cdf(self, points=None):
        """
        A Studentst cumulative density function.

        :param Studentst self:
            An instance of the Student's T class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Student's T distribution.
        :return:
            Cumulative density values along the support of the Student's T distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_icdf(self, xx):
        """
        A Chi-squared inverse cumulative density function.

        :param Studentst self:
            An instance of Student's T class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Student's T distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m=None):
        """
        Generates samples from the Student's T distribution.

        :param Studentst self:
            An instance of Student's T class
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e05 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)
