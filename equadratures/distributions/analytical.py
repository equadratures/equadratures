""" The Analytical distribution"""
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
from equadratures.distributions.template import Distribution
import numpy as np
import scipy.stats as stats
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
RECURRENCE_PDF_SAMPLES = 50000

class Analytical(Distribution):
    """ The class defines a Analytical object.

        :param Weight weight_function:
              An instance of the Weight class.
    """
    def __init__(self, weight_function):
        self.weight_function = weight_function
        self.data = weight_function.data
        self.lower = weight_function.lower
        self.upper = weight_function.upper
        self.mean = weight_function.mean
        self.variance = weight_function.variance
        self.bounds = weight_function.support
        self.x_range_for_pdf = weight_function.x_range_for_pdf

    def get_description(self):
        """ A destription of Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :return:
                A string describing the Analytical distribution.
        """
        text = "is a Analytical distribution defined over a support from "+str(self.lower)+" to "+str(self.upper)+". \
            It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
        return text

    def get_pdf(self, points=None):
        """ A Analytical probability density function.

            :param Analytical self:
                An instance of Analytical class.
            :param points:
                An array of points in which the probability density function needs to be calculated.
            :return:
                Probability density values along the support of Analytical distribution.
            ** Notes **
            To obtain a probability density function from finite samples, this function uses kerne density estimation (with Gaussian kernel).
        """
        return self.weight_function.get_pdf(points)

    def get_cdf(self, points=None):
        y = self.get_pdf()
        summ = np.sum(y)
        p = np.array(y/summ)
        analytical = stats.rv_discrete(name='analytical', values=(self.x_range_for_pdf, p))
        return analytical.cdf(points)

    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the Analytical distribution.

        :param Analytical self:
            An instance of Analytical class.
        :param array order:
            The order of the recurrence coefficients desidered.
        :return:
            Recurrence coefficients associated with the Analytical distribution.
        """
        #x = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        w = self.data

        # Allocate memory for recurrence coefficients
        order = int(order)+1
        w = w / np.sum(w)
        ab = np.zeros((order+1,2))

        # Negate "zero" components
        nonzero_indices = []
        for i in range(0, len(w)):
            if w[i] != 0:
                nonzero_indices.append(i)

        ncap = len(nonzero_indices)
        x = self.x_range_for_pdf[nonzero_indices] # only keep entries at the non-zero indices!
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
        A Analytical inverse cumulative distribution function.

        :param Analytical self:
            An instance of Analytical class.
        :param array xx:
            An array of points in which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Analytical distribution.
        """
        y = self.get_pdf()
        summ = np.sum(y)
        p = np.array(y/summ)
        analytical = stats.rv_discrete(name='analytical', values=(self.x_range_for_pdf, p))
        return analytical.ppf(xx)
