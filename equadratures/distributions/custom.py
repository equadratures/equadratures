""" The Custom distribution"""
import numpy as np
from distribution import Distribution
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from recurrence_utils import jacobi_recurrence_coefficients
import scipy.stats as stats
RECURRENCE_PDF_SAMPLES = 8000

#-----------------#
import matplotlib.pyplot as plt
#-----------------#
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
             self.std      = np.std(self.data)
             self.lower    = min(self.data)
             self.upper    = max(self.data)
             self.bounds   = np.array([self.lower, self.upper])
             self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
             self.skewness = stats.skew(self.data)
             self.kurtosis = stats.kurtosis(self.data) 
        
    def getDescription(self):
        """ A destription of custom distribution.
            
            :param Custom self:
                An instance of Custom class.
            :return:
                A string describing the Custom distribution.
        """
        text = "A Custom distribution has been defined over a support from "+str(self.lower)+" to "+str(self.upper)+". It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
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
        X = np.array(points)
        kernel = stats.gaussian_kde(self.data)
        if points is not None: 
            # check dimensions:
            points = np.matrix(points)
            dimension = np.shape(points)
            summ = dimension[0]+dimension[1]
            if (summ != 2) :
                wts    = kernel(X)
                return wts
            else:
                c = X
                lower = c*(1.-c/1000.)
                upper = c*(1.+c/1000.)
                vector = np.linspace(lower, upper, 3)
                wts_v  = kernel(vector)
                wts    = wts_v[1]
                return wts
        else:
            print 'An input array have to be given to the getPDF method.'

    #------------------------------------------------------------------------#
    def getCDF(self, points=None):
        points = np.matrix(points)

        y = self.getPDF(self.data) 
        summ = np.sum(y) 
        p = np.array(y/summ)
        custom = stats.rv_discrete(name='custom', values=(self.data, p)) 

        return custom.cdf(points)
        #------------------------------------------------------------------------#
            
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
        xx = np.matrix(xx)
        y = self.getPDF(self.data)
        summ = np.sum(y)
        p = np.array(y/summ)
        custom = stats.rv_discrete(name='custom', values=(self.data, p))
        return custom.ppf(xx)

