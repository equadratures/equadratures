from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


def fun(x):
    return np.sum(x)

def fun2(x):
    return np.exp(x[0] + x[1])

class TestPolylsq(TestCase):

    def test_polylsqA(self):
        dimensions = 20
        zeta_1 = Parameter(distribution='gaussian', shape_parameter_A = 10.0, shape_parameter_B=0.5, order=1)
        myParams = [zeta_1 for i in range(dimensions)]
        myBasis = Basis('Total order')
        myPoly = Polylsq(parameters=myParams, basis=myBasis, mesh='tensor', optimization='random', oversampling=1.5)
        myPoly.computeCoefficients(fun)
        stats = myPoly.getStatistics(light=True, max_sobol_order=1)
        np.testing.assert_almost_equal(stats.mean, 200.0, decimal=10, err_msg = "Difference greated than imposed tolerance for mean value")
        np.testing.assert_almost_equal(stats.variance, 10.0, decimal=10, err_msg = "Difference greated than imposed tolerance for mean value")

    def test_polylsqB(self):
        dimensions = 2
        zeta_1 = Parameter(distribution='uniform', lower=-1., upper=1., order=25)
        myParams = [zeta_1 for i in range(dimensions)]
        myBasis = Basis('Tensor grid')
        myPoly = Polylsq(parameters=myParams, basis=myBasis, mesh='tensor', optimization='greedy-qr', oversampling=1.0)
        myPoly.computeCoefficients(fun)
        
        tensor = Basis('Tensor grid')
        myPolyTensor = Polyint([zeta_1, zeta_1], tensor)
        myPolyTensor.computeCoefficients(fun)
        np.testing.assert_almost_equal(myPoly.coefficients, myPolyTensor.coefficients, decimal=13, err_msg = "Difference greated than imposed tolerance for mean value")

if __name__== '__main__':
    unittest.main()