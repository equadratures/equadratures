from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


def fun(x):
    return np.exp(3*x[0] + x[1])

class TestSparse(TestCase):

    def test_spam(self):
        x = Parameter(distribution='uniform', lower=-1., upper=1., order=4)
        sparse = Basis('Sparse grid', level=6, growth_rule='exponential')
        myPolySparse = Polyint([x,x], sparse)
        myPolySparse.computeCoefficients(fun)

        x = Parameter(distribution='uniform', lower=-1., upper=1., order=35)
        tensor = Basis('Tensor grid')
        myPolyTensor = Polyint([x,x], tensor)
        myPolyTensor.computeCoefficients(fun)

        c1 = myPolySparse.coefficients
        c2 = myPolyTensor.coefficients

        np.testing.assert_almost_equal(float(c1[0]), float(c2[0]), decimal=7, err_msg = "Difference greated than imposed tolerance for mean value")

    def test_spam2(self):
        x = Parameter(distribution='uniform', lower=-1., upper=1., order=4)
        sparse = Basis('Sparse grid', level=7, growth_rule='exponential')
        myPolySparse = Polyint([x,x], sparse)
        multi_indices = sparse.elements
        np.testing.assert_almost_equal(np.max(multi_indices), 128, decimal=7, err_msg = "Difference greated than imposed tolerance for mean value")
                                          
if __name__== '__main__':
    unittest.main()
