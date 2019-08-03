from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


class TestSavePoly(TestCase):
    '''
    I am going to test Poly_Minimal and convert2Full/convert2Min here. I am not sure Travis plays well with
    testing routines that save/load files from a directory.
    '''
    def test_conversion(self):
        d = 2
        param = Parameter(distribution='Uniform', lower=-1, upper=1., order=1)
        myParameters = [param for _ in range(d)]
        def f(x):
            return x[0] * x[1]

        np.random.seed(1)
        x_train = np.random.randn(100,2)
        polynomialOrders = np.full(d, 2)
        myBasis = Basis('Tensor grid', polynomialOrders)
        poly = Polyreg(myParameters, myBasis, training_inputs=x_train, fun=f)
        poly_min = poly.convert2Min()

        poly_full = Poly.convert2Full(poly_min)

        np.testing.assert_array_equal(poly.coefficients, poly_full.coefficients)
        np.testing.assert_array_equal(poly.quadraturePoints, poly_full.quadraturePoints)
        np.testing.assert_array_equal(poly.quadratureWeights, poly_full.quadratureWeights)

if __name__ == '__main__':
    unittest.main()
