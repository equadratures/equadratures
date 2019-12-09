from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

def fun(x):
     return 5.0 * x[0]**3 - x[0]*x[1] + 3.0*x[1]*x[2]**3 + 32.0

class TestF(TestCase):
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
          myBasis = Basis('tensor-grid')
          myPoly = Poly([zeta_1, zeta_2, zeta_3], myBasis, method='numerical-integration')
          myNataf = Correlations(myPoly, R)
          samples_mc = myNataf.get_correlated_samples(N=50000)
          f_mc = evaluate_model(samples_mc, fun)
          samples_p = myNataf.set_model(fun)
          myTransformedPoly = myNataf.get_transformed_poly()
          mean, variance = myTransformedPoly.get_mean_and_variance()
          skewness, kurtosis = myTransformedPoly.get_skewness_and_kurtosis()
          np.testing.assert_almost_equal(np.mean(f_mc)*0.01, mean*0.01, decimal=1, err_msg = "Difference greated than imposed tolerance")
          np.testing.assert_almost_equal(np.var(f_mc)*0.000001, variance*0.000001, decimal=2, err_msg = "Difference greated than imposed tolerance")
          np.testing.assert_almost_equal( skew(f_mc)*0.1, skewness*0.1, decimal=1, err_msg = "Difference greated than imposed tolerance")

if __name__== '__main__':
    unittest.main()