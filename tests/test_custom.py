from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt

class Test_Custom(TestCase):
    """ This class compares the probabilitity density functions,
        the cumulative distribution functions, the inverse of 
        the cdf obtained using the related methods of the Custom
        class, given a set of points which belong to a known
        distribution.
    """
    D = list() # list of object that will be tested
    D.append(Parameter(order = 3 , distribution='gaussian', shape_parameter_A = 0.0, shape_parameter_B=1.0))
    D.append(Parameter(order=5, distribution='uniform', lower=-1., upper =1.))
    D.append(Parameter(order=5, distribution='rayleigh', shape_parameter_A =1.))
    D.append(Parameter(order=5, distribution='beta', shape_parameter_A = 1., shape_parameter_B = 1., lower=0., upper = 1.))
    D.append(Parameter(order=5, distribution='Chebyshev', upper = 1., lower=0.))
    D.append(Parameter(order=5, distribution='Chisquared', shape_parameter_A = 14))
    D.append(Parameter(order=5, distribution='exponential', shape_parameter_A = 0.7))
    D.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A = 1., shape_parameter_B = 1., lower = 0.5, upper = 1.5))

    for i in range(len(D)):
       data1 = D[i].getSamples(m=500) 
       data1 = np.sort(data1)
       custom = Parameter(order=3, distribution='custom', data=data1)

       # pdf :

       pdf  = custom.getPDF(data1)
       g_pdf = D[i].getPDF(data1)

       plt.figure()
       plt.grid(linewidth = 0.4, color = 'k')
       plt.plot(data1, pdf, 'bo', label ='pdf of Custom')
       plt.plot(data1, g_pdf, 'ro', label=D[i].name)
       plt.legend(loc='lower center')
       plt.show()

    for i in range(len(D)):
        data1 = D[i].getSamples(m=500)
        data1 = np.sort(data1)
        custom = Parameter(order=3, distribution='custom', data=data1)
                        
        cdf  = custom.getCDF(data1)
        g_cdf = D[i].getCDF(data1)

        plt.figure()
        plt.grid(linewidth = 0.4, color = 'k')
        plt.plot(data1, cdf, 'bo', label='cdf of Custom Class')
        plt.plot(data1, g_cdf, 'ro', label= D[i].name)
        plt.legend(loc='upper left')
        plt.show()

    for i in range(len(D)):
        data1 = D[i].getSamples(m=500) 
        custom = Parameter(order=3, distribution='custom', data=data1)
        
        icdf = custom.getiCDF(cdf)
        g_icdf = D[i].getiCDF(g_cdf)
        
        plt.figure()
        plt.grid(linewidth = 0.4, color = 'k')
        plt.plot(cdf, icdf, 'bo', label='icdf of Custom Class')
        plt.plot(g_cdf, g_icdf, 'ro', label= D[i].name)
        plt.legend(loc='upper left')
        plt.show()


if __name__== '__main__':
     unittest.main()

