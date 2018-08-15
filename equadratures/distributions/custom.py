""" The Custom distribution"""
import numpy as np
from distribution import Distribution
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from recurrence_utils import jacobi_recurrence_coefficients
import scipy.stats as stats
RECURRENCE_PDF_SAMPLES = 8000

class Custom(Distribution):
    """ The class defines a Custom object.
            
        :param data:
              A numpy array with data values (x-y column format). Note this option is only invoked if the user uses the Custom param_type.
    """
    def __init__(self, data):
        if data is not None:
             self.data     = data
             self.mean     = np.mean(self.data)
             self.variance = np.var(self.data)
             self.lower    = np.min(self.data)
             self.upper    = np.max(self.data)
             # the following lines are correct?
             self.bounds   = np.array([self.lower, self.upper])
             self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
             self.skewness = stats.skew(self.data)
             self.kurtosis = stats.kurtosis(self.data)
             #else:     
             #raise(ValueError, 'Custom class __init__: if data is provided then the custom distribution must be selected!')
        
    def getDescription(self):
        """ A destription of custom distribution.
            
            :param Custom self:
                An instance of Custom class.
            :return:
                A string describing the Custom distribution.
        """
        text = "A Custom distribution has been defined over a suppor from "+str(self.lower)+" to "+str(self.upper)+". It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
        return text
    
    def getPDF(self, points=None):
        """ A custom probability density function.
            
            :param Custom self:
                An instance of Custom class.
            :param points:
                An array of points in which the probability density function needs to be calculated.
            :return:
                Probability density values along the support of custom distribution.
            ** Notes **
            To obtain a probability density function from finite samples, this function uses kerne density estimation (with Gaussian kernel).
        """
        if points is not None:
            kernel = stats.gaussian_kde(self.data)
            wts    = kernel(points)
            return wts
        else:
            print 'An input array have to be given to the getPDF method.'

    def getCDF(self, points=None):
        """ A cumulative density function associated with a given data set.
            
            :param points:
                An array of points in which the cumulative distribution function needs to be evaluated.
            :return:
                Cumulative distribution function values along the support of the custom distribution.
        """
        if points is not None:
            x = points
            y = self.getPDF(self.data)
            c = []
            c.append(0.0)
            for i in range(1, len(x)):
                c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
            for i in range(1,len(x)):
                c[i] = c[i]/c[len(x)-1]
            return c
        else:
            print 'An input array has to be given to the getCDF method.'
            
    def getRecurrenceCoefficients(self, order):
        """
        Recurrence coefficients for the custom distribution.

        :param Custom self:
            An instance of Custom class.
        :param array order:
            The order of the recurrence coefficients desidered.
        :return:
            Recurrence coefficients associated with the custom distribution.
        """
        print 'this method has to be completed!'

    def getiCDF(self, xx):
        """ 
        A custom inverse cumulative distribution function.
        
        :param Custom self:
            An instance of Custom class.
        :param array xx:
            An array of points in which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Custom distribution.
        """
        x  = self.data
        y  = self.getPDF(x)
        c  = []
        yy = []
        c.append(0.0)
        for i in range(1, len(x)):
            c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
        for i in range(1, len(x)):
            c[i]=c[i]/c[len(x)-1]
        for k in range(0, len(x)):
            for i in range(0, len(x)):
                if ((xx[k]>=c[i]) and (xx[k]<=c[i+1])):
                    value = float((xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i])
                    yy.append(value)
                    break
        return yy



