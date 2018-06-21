from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from math import exp

class TestQR(TestCase):

    def testbasic(self):
        # blackbox
        def blackbox(x):
            return x

        #-----------------------------------------------#
        # analytical distributions 
        # 1) Arcsine distribution --> Chebyshev

        # support ---> a < b , -\infty. +\infy
        a = 0.0#0.001
        b = 1.0#0.99
        x = np.linspace(a, b, 100) # domain for Chebyshev
        mean_1 = (a+b)/2.0 
        variance_1 = (1.0/8.0)*(b-a)**2

        f_X= np.zeros(len(x))

        for i in range(0,len(x)):
            if x[i] == a :
                f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
            elif x[i] == b:
                f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
            else: 
                f_X[i] = 1.0/(np.pi* np.sqrt((x[i] - a)*(b - x[i])) )

        #print f_X
        print 'analytical mean of arcsine:', mean_1
        print 'analytical variance of arcsine:', variance_1

        #----------- effective quadrature -----------------#

        xo = Parameter(order=5, distribution='Chebyshev',lower =0.001, upper=0.99)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean
        print 'Effective quadrature variance:' , myStats.variance
        print 'done!'

if __name__ == '__main__':
    unittest.main()
