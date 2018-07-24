"""The Beta distribution."""
import numpy as np
from distribution import Distribution
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from recurrence_utils import jacobi_recurrence_coefficients

class Beta(Distribution):
    """
    The class defines a Beta object. It is the child of Distribution.
    
    :param double a:
        First shape parameter of the beta distribution. This value has to be greater than 0.
    :param double b:
            Second shape parameter of the beta distribution. This value has to be greater than 0.
    :param double lower:
        Lower bound of the support of the beta distribution.
    :param double upper:
        Upper bound of the support of the beta distribution.
    """
    def __init__(self, lower=None, upper=None, shape_A=None, shape_B=None):
        self.shape_A = shape_A
        self.shape_B = shape_B
        self.lower = lower
        self.upper = upper
        if self.shape_A >= 1. and self.shape_B >= 1.0:
            self.mean = (self.shape_A) / (self.shape_A + self.shape_B)
            self.variance = (self.shape_A * self.shape_B) / ( (self.shape_A + self.shape_B)**2 * (self.shape_A + self.shape_B + 1.0) )
            self.skewness = 2.0 * (self.shape_B - self.shape_A) * np.sqrt(self.shape_A + self.shape_B + 1.0) / ( (self.shape_A + self.shape_B + 2.0) * np.sqrt(self.shape_A * self.shape_B) ) 
            self.kurtosis = 6.0 * ((self.shape_A - self.shape_B)**2 * (self.shape_A + self.shape_B + 1.0) - self.shape_A * self.shape_B * (self.shape_A + self.shape_B + 2.0)  ) /( (self.shape_A * self.shape_B) * (self.shape_A + self.shape_B + 2.0) * (self.shape_A + self.shape_B + 3.0)) + 3.0   

        self.bounds = np.array([0, 1])

    
    def getDescription(self):
        """
        A description of the beta distribution.
            
        :param Beta self:
            An instance of the beta class.
        :return:
            A string describing the beta distribution.
        """
        text = "A beta distribution is defined over a support; given here as "+str(self.lower)+", to "+str(self.upper)+". It has two shape parameters, given here to be "+str(self.shape_A)+" and "+str(self.shape_B)+"."
        return text

    def getPDF(self, N=None, points=None):
        """
        A beta probability density function.
        
        :param Beta self:
            An instance of the Beta class.
        :param integer N:
            Number of points for defining the probability density function.
        :return:
            Probability density values along the support of the Beta distribution.
        """
        if N is not None:
            x = np.linspace(0, 1, N)
            w = (x**(self.shape_A - 1) * (1 - x)**(self.shape_B - 1))/(beta(self.shape_A, self.shape_B) )
            xreal = np.linspace(self.lower, self.upper, N)
            wreal = w * (1.0)/(self.upper - self.lower)
            return xreal, wreal
        elif points is not None:
            w = (points**(self.shape_A - 1) * (1 - points)**(self.shape_B - 1))/(beta(self.shape_A, self.shape_B) )
            wreal = w * (1.0)/(self.upper - self.lower)
            return wreal
        else:
            raise(ValueError, 'Please specify an input for getPDF method')

    def getCDF(self, N=None, points=None):
        """
        A beta cumulative density function.
        
        :param Beta self:
            An instance of the Beta class.
        :param integer N:
            Number of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the Gamma distribution.
        """
        if N is not None:
            xreal = np.linspace(self.lower, self.upper, N)
            x = np.linspace(0, 1, N)
            w = np.zeros((N,1))
            for i in range(0, N):
                w[i] = betainc(a, b, x[i])
            return xreal, w
        elif points is not None:
            t = points.T
            for i in range(len(points)):
                for j in range(len(t)):
                    w[i,j] = betainc(a,b, points[i,j])
            return w

    def getRecurrenceCoefficients(self, order):
        """
        Recurrence coefficients for the beta distribution.
        
        :param Beta self:
            An instance of the Beya class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the beta distribution.
        """
        ab =  jacobi_recurrence_coefficients(self.shape_B - 1.0, self.shape_A - 1.0, self.lower, self.upper, order)
        return ab
