from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import numexpr as ne

expr = "x**3 - 2*x**2"
def f(x):
    return ne.evaluate(expr)

class TestPolyroots(TestCase):
    def test_chebshev_roots(self):
        order = 5
        param = Chebyshev(order=order, low=-1, up=1)

        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(f)

        roots = myPoly.get_poly_roots()

        np.testing.assert_almost_equal(roots[0], 0, decimal=6)

if __name__== '__main__':
    unittest.main()
