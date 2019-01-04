from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

def fun(x):
     return 5.0 * x[0]**3 - x[0]*x[1] + 3.0*x[1]*x[2]**3 + 32.0

class Test_Nataf(TestCase):
     def test_nataf(self):
          zeta_1 = Parameter(distribution='truncated-gaussian', shape_parameter_A = 3.0, shape_parameter_B = 2.0, order=3, lower=-2.0, upper=4.0)
          zeta_2 = Parameter(distribution='truncated-gaussian', shape_parameter_A = -1.0, shape_parameter_B = 0.1, order=3, lower=-5.0, upper=5.0)
          zeta_3 = Parameter(distribution='truncated-gaussian', shape_parameter_A = 2.0, shape_parameter_B = 2.0, order=3, lower=0.0, upper=6.0)
          R = np.eye(3)
          R[0, 1] = 0.6
          R[0, 2] = 0.3            
          R[2, 1] = 0.2
          R[1, 0] = R[0, 1]
          R[2, 0] = R[0, 2]
          R[1, 2] = R[2, 1]   

          u1 = Parameter(distribution='normal', shape_parameter_A=0.0, shape_parameter_B=1.0, order=3)
          myNataf = Nataf([zeta_1, zeta_2, zeta_3], R)

          # For Monte-Carlo!
          samples_mc = myNataf.getCorrelatedSamples(N=50000)
          f_mc = evalfunction(samples_mc, fun)

          # For Polynomials!
          myBasis = Basis('Tensor grid')
          myPoly = Polyint([u1, u1, u1], myBasis)
          samples_p =  myPoly.quadraturePoints
          samples_corr_p = myNataf.U2C(samples_p)
          f_p = evalfunction(samples_corr_p, fun)

          myPoly.computeCoefficients(f_p)
          myStats = myPoly.getStatistics()

          print '----MONTE CARLO----'
          print np.mean(f_mc), np.var(f_mc), skew(f_mc)

          print '----POLYNOMIALS-----'
          print myStats.mean, myStats.variance, myStats.skewness      

          np.testing.assert_almost_equal(np.mean(f_mc)*0.01, myStats.mean*0.01, decimal=1, err_msg = "Difference greated than imposed tolerance")
          np.testing.assert_almost_equal(np.var(f_mc)*0.000001, myStats.variance*0.000001, decimal=2, err_msg = "Difference greated than imposed tolerance")
          np.testing.assert_almost_equal( skew(f_mc)*0.1, myStats.skewness*0.1, decimal=1, err_msg = "Difference greated than imposed tolerance")

if __name__== '__main__':
    unittest.main()
