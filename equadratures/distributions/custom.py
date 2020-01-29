""" The Custom distribution"""
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
from equadratures.distributions.template import Distribution
import numpy as np
import scipy.stats as stats
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
RECURRENCE_PDF_SAMPLES = 50000

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
    def get_description(self):
        """ A destription of custom distribution.

            :param Custom self:
                An instance of Custom class.
            :return:
                A string describing the Custom distribution.
        """
        text = "is a Custom distribution defined over a support from "+str(self.lower)+" to "+str(self.upper)+". It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
        return text
    def get_pdf(self, points=None):
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
            points = np.array([points])
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
            print('An input array have to be given to the getPDF method.')

    """def getCDF(self, points=None):
        # A cumulative density function associated with a given data set.

            :param points:
                An array of points in which the cumulative distribution function needs to be evaluated.
            :return:
                Cumulative distribution function values along the support of the custom distribution.
        #
        if points is not None:
            x = sorted(points)
            #y = self.getPDF(self.data)
            y = self.getPDF(x)

            c = []
            c.append(0.0)
            for i in range(1, len(x)):
                c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
            for i in range(1,len(x)):
                c[i] = c[i]/c[len(x)-1]
            return c
        else:
            print 'An input array has to be given to the getCDF method.'
    """
    #------------------------------------------------------------------------#
    def get_cdf(self, points=None):
        # Approssimation of PDF integral (not into the original version of custom class)
        #    given a set of points: what is the cdf, staring from the PDF of data?
        #------------------------------------
        # version 1:
        #---------------------------------------------------------------------
        #x = sorted(points) # points can be different from self.data
        #
        #y = self.getPDF(x) # pdf function associated with self.data and points
        #
        #c     = [] # list for future CDF array
        #c.append( 0.) # initialization
        #
        #for i in range(1,len(points)):
        #    c.append((y[i-1]+y[i] )*.5*(x[i]-x[i-1])+c[i-1] )
        #for i in range(1,len(points)):
        #    c[i] = c[i]/c[len(points)-1]
        #return c
        #--------------------------------------------------------------------
        # version 2
        points = np.array(points)

        y = self.get_pdf(self.data)
        summ = np.sum(y)
        p = np.array(y/summ)
        custom = stats.rv_discrete(name='custom', values=(self.data, p))

        return custom.cdf(points)
        #------------------------------------------------------------------------#
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the custom distribution.

        :param Custom self:
            An instance of Custom class.
        :param array order:
            The order of the recurrence coefficients desidered.
        :return:
            Recurrence coefficients associated with the custom distribution.
        """
        x = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        w = self.get_pdf(points = x)
        # Allocate memory for recurrence coefficients
        order = int(order)+1
        w = w / np.sum(w)
        ab = np.zeros((order+1,2))

        # Negate "zero" components
        nonzero_indices = []
        for i in range(0, len(x)):
            if w[i] != 0:
                nonzero_indices.append(i)

        ncap = len(nonzero_indices)
        x = x[nonzero_indices] # only keep entries at the non-zero indices!
        w = w[nonzero_indices]
        s = np.sum(w)

        temp = w/s
        ab[0,0] = np.dot(x, temp.T)
        ab[0,1] = s


        if order == 1:
            return ab

        p1 = np.zeros((1, ncap))
        p2 = np.ones((1, ncap))

        for j in range(0, order):
            p0 = p1
            p1 = p2
            p2 = ( x - ab[j,0] ) * p1 - ab[j,1] * p0
            p2_squared = p2**2
            s1 = np.dot(w, p2_squared.T)
            inner = w * p2_squared
            s2 = np.dot(x, inner.T)
            ab[j+1,0] = s2/s1
            ab[j+1,1] = s1/s
            s = s1

        return ab
    def get_icdf(self, xx):
        """
        A custom inverse cumulative distribution function.

        :param Custom self:
            An instance of Custom class.
        :param array xx:
            An array of points in which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Custom distribution.
        """
        #x  = self.data
        #y  = self.getPDF(x)
        #c  = []
        #yy = []
        #c.append(0.0)
        #for i in range(1, len(x)):
        #    c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
        #for i in range(1, len(x)):
        #    c[i]=c[i]/c[len(x)-1]
        #for k in range(0, len(x)):
        #    for i in range(0, len(x)):
        #        if ((xx[k]>=c[i]) and (xx[k]<=c[i+1])):
        #            value = float((xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i])
        #            yy.append(value)
        #            break
        #return yy
        xx = np.array(xx)
        y = self.get_pdf(self.data)
        summ = np.sum(y)
        p = np.array(y/summ)
        custom = stats.rv_discrete(name='custom', values=(self.data, p))
        return custom.ppf(xx)