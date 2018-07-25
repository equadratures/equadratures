"""The Truncated Gaussian distribution."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from distribution import Distribution
from gaussian import *

class TruncatedGaussian(Distribution):
    """
    The class defines a Truncated-Gaussian object. It is the child of Distribution.
    :param double mean:
		Mean of the truncated Gaussian distribution.
	:param double variance:
		Variance of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the truncated Gaussian distribution.
    """
    def __init__(self, mean, variance, lower, upper):
        if (mean is not None) and (variance is not None) and (lower is not None) and (upper is not None):
            meanParent = mean
            varianceParent = variance
            self.std    = Gaussian(mean=0.0, variance=1.0)
            self.parent = Gaussian(mean= meanParent, variance=varianceParent)
            self.lower = lower 
            self.upper = upper
            self.skewness = 0.0
            self.kurtosis = 0.0
            self.bounds = np.array([-np.inf, np.inf]) 
            self.beta  = (self.upper - self.parent.mean)/self.parent.variance  
            self.alpha = (self.lower - meanParent)/varianceParent
            num = self.std.getPDF(points=self.beta)-self.std.getPDF(points=self.alpha)
            den = self.std.getCDF(points=self.beta)-self.std.getCDF(points=self.alpha)
            self.mean = meanParent - varianceParent*(num/den)
            
            num_i = self.beta*self.std.getPDF(points=self.beta)-self.alpha*self.std.getPDF(points=self.alpha)
            den   = self.std.getCDF(points=self.beta)-self.std.getCDF(points=self.alpha)
            num_ii= self.std.getPDF(points=self.beta)-self.std.getPDF(points=self.alpha)
            self.variance = varianceParent*(1-(num_i/den)-(num_ii/den)**2)
            self.sigma = np.sqrt(self.variance)

    def getDescription(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = "A truncated Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+", and a lower bound of "+str(self.lower)+" and an upper bound of "+str(self.upper)+"."
        return text

    def getPDF(self, N=None, points=None):
        """
        A truncated Gaussian probability distribution.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
		:param integer N:
            Number of equidistant points over the support of the distribution; default value is 500.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the truncated Gaussian distribution.
        """
        if N is not None:
            x = np.linspace(self.lower, self.upper, N)
            num = self.parent.getPDF(points = x)
            den = self.parent.getCDF(points = self.upper)-self.parent.getCDF(points =self.lower)
            w = num/den
            return x,w

        elif points is not None:
            num = self.parent.getPDF(points = points)
            den = self.parent.getCDF(points = self.upper)-self.parent.getCDF(points =self.lower)
            w = num/den
            return w        

    def getCDF(self, N=None, points = None):
        """
        A truncated Gaussian cumulative density function.

	    :param truncated Gaussian self:
            An instance of the Gaussian class.
        :param integer N:
            Number of points for defining the cumulative density function; default value is 500.
        :return:
            An array of N equidistant values over the support of the truncated Gaussian.
        :return:
            Gaussian cumulative density values.
        """
        if N is not None:
            x = np.linspace(self.lower, self.upper, N)
            num = self.parent.getCDF(points=x) - self.parent.getCDF(points=self.lower)
            den = self.parent.getCDF(points=self.upper) - self.parent.getCDF(points=self.lower)
            w = num/den
            return x,w
        elif points is not None:
            num = self.parent.getCDF(points=points) - self.parent.getCDF(points=self.lower)
            den = self.parent.getCDF(points=self.upper) - self.parent.getCDF(points=self.lower)
            w = num/den
            return w

    """
    def getiCDF(self, xx):
        
        num = self.parent.getCDF(points=xx) - self.parent.getCDF(points=self.lower)
        den = self.parent.getCDF(points=self.upper) - self.parent.getCDF(points=self.lower)
        p = num / den 

        #print 'hello from getiCDF of gaussian!'
        #print  'p = num/den = ', p
        #pp = self.parent.getCDF(points=self.lower)+ p
        #print 'parent is using CDF and its result is:', pp
        w = self.parent.getiCDF(p)
        #print 'parent is using iCDF: the results are:',  w
        return w
    """