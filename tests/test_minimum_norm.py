from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
def fun(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
class TestJ(TestCase):
    def test_qr(self):
        zeta_1 = Parameter(distribution='uniform', order=4, lower= -2.0, upper=2.0)
        zeta_2 = Parameter(distribution='cauchy', order=4, shape_parameter_A=3.2, shape_parameter_B=1.2)
        myBasis1 = Basis('tensor-grid')
        myBasis2 = Basis('total-order')
        myPoly1 = Poly([zeta_1, zeta_2], myBasis1, method='numerical-integration')
        myPoly2 = Poly([zeta_1, zeta_2], myBasis2, method='minimum-norm', sampling_args={'mesh':'monte-carlo', 'subsampling-algorithm':'svd', 'sampling-ratio':0.99})
        myPoly1.set_model(fun)
        myPoly2.set_model(fun)
        c1 = myPoly1.get_coefficients()
        c2 = myPoly2.get_coefficients()
        np.testing.assert_array_almost_equal(c1[0], c2[0], decimal=3, err_msg='Problem!')
if __name__== '__main__':
    unittest.main()