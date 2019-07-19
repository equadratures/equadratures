"""The Chi-squared distribution."""

from equadratures.distributions.template import Distribution

import numpy as np
from scipy.special import erf, erfinv, gamma, gammainc
from scipy.stats import chi2

RECURRENCE_PDF_SAMPLES = 8000

class Chisquared(Distribution):
    """
    The class defines a Chi-squared object. It is the child of Distribution.
    
    :param int dofs:
		Degrees of freedom for the chi-squared distribution.
    """
    def __init__(self, dofs):
        self.dofs = dofs
        if self.dofs is not None:
            if self.dofs == 1:
                self.bounds = np.array([1e-15, np.inf])
            else:
                self.bounds = np.array([0.0, np.inf])
            if self.dofs >= 1:
                self.mean = float(self.dofs)
                self.variance = 2 * self.mean
                self.skewness = np.sqrt(8.0 / self.mean)
                self.kurtosis = 12.0/self.mean + 3.0
                self.x_range_for_pdf = np.linspace(0.0, 10.0*self.mean,RECURRENCE_PDF_SAMPLES)
                self.parent = chi2(self.dofs)
    
    def getDescription(self):
        """
        A description of the Chi-squared distribution.
            
        :param Chi-squared self:
            An instance of the Chi-squared class.
        :return:
            A string describing the Chi-squared distribution.
        """
        text = "A Chi-squared distribution is characterised by its degrees of freedom, which here is"+str(self.dofs)+"."
        return text

    def getPDF(self, points=None):
        """
        A Chi-squared  probability density function.
        
        :param Chi-squared  self:
            An instance of the Chi-squared  class.
        :param points:
            Matrix of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Chi-squared distribution.
        :return:
            Probability density values along the support of the Chi-squared distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else: 
            raise(ValueError, 'Please digit an input for getPDF method')

    def getCDF(self, points=None):
        """
        A Chi-squared cumulative density function.
        
        :param Chi-squared self:
            An instance of the Chi-squared class.
        :param matrix points:
            Matrix of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Chi-squared distribution.
        :return:
            Cumulative density values along the support of the Chi-squared distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise(ValueError, 'Please digit an input for getCDF method')


    def getiCDF(self, xx):
        """
        A Chi-squared inverse cumulative density function.

        :param Chi2:
            An instance of Chi-squared class
        :param matrix xx:
            A matrix of points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Chi-squared distribution.
        """
        return self.parent.ppf(xx)
    
    def getSamples(self, m=None):
        """ 
        Generates samples from the Chi-squared distribution.

        :param chi2 self:
            An instance of Chi-squared class
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

