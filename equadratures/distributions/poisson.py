# -*- coding: utf-8 -*-
"""The Poisson Distribution"""

from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.stats import poisson

RECURRENCE_PDF_SAMPLES = 8000

class Poisson(Distribution):
    """
     The class defines a 
     Poisson object. It is a child of Distribution.
     
     :param double mean:
         Mean of the poisson Distribution.
     :param double a:
         Shape parameter for poisson Distribution
    """
    
    def __init__(self ,shape_A):
        self.shape_A=shape_A
        if self.shape_A is not None :
            if self.shape_A < 0 :
                raise ValueError("Mean of poisson distribution should be positive.")
            self.parent=poisson(mu=self.shape_A)
            self.mean,self.variance,self.skewness,self.kurtosis = self.parent.stats(moments='mvsk')
            self.x_range_for_pdf=np.linspace(0 , self.shape_A , RECURRENCE_PDF_SAMPLES)
           
           
    

    def get_description(self):
        """
        A description of the poisson distribution.
        
        :param Poisson self:
            An instance of the Poisson class.
        :return:
            A string describing the poisson distribution.
        """           
        text = "is a poisson distribution over the support "+str(self.shape_A)+"as mean or mu."
        return text

    def get_cdf(self,points=None):
        """
        A poisson cumulative density function.
        
        :param poisson self:
            An instance of the poisson class
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative Density values along the support of the poisson distribution.
        """
        if points is not None:
            return self.parent.cdf(points)
        else:
            raise ValueError('Please digt an input for getCDF method')
     
    
    def get_pdf(self,points=None):
        """
        A poisson probability density function.
        
        :param poisson self:
            An instance of the poisson class
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability Density values along the support of the poisson distribution.
        """
        if points is not None:
            return self.parent.pdf(points)
        else:
            raise ValueError('Please digit an input for getPDF method')
    

    def get_icdf(self, xx):
        """
        A Poisson inverse cumulative density function.
        :param Poisson self:
            An instance of Poisson class.
        :param array xx:
            Points at which the inverse cumulative density funcion needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Poisson distribution.
        """
        return self.parent.ppf(xx)

    def get_samples(self, m =None):
        """ Generates samples from the Poisson distribution.
            :param Poisson self:
                An instance of Poisson  class.
            :param integer m:
                Number of random samples. If no provided, a default value of 5e5 is assumed.
            :return:
                A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size= number)

