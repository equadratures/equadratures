from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestBasis(TestCase):

    def test_tensor(self):
        tensor = Basis('Tensor grid', [4, 4, 4])
        np.testing.assert_almost_equal(tensor.cardinality, 125, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_sparse(self):  
        sparse = Basis('Sparse grid', level=2, growth_rule='linear')
        sparse.dimension = 3
        a, b, c = sparse.getBasis()
        np.testing.assert_almost_equal(31, len(a), decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_euclidean(self):
        euclid = Basis('Euclidean degree', [4, 4, 4])
        np.testing.assert_almost_equal(euclid.cardinality, 54, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_hyperbolic(self):
        hyper = Basis('Hyperbolic basis', [4, 4, 4], q=0.5)
        np.testing.assert_almost_equal(hyper.cardinality, 16, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_total(self):
        hyper = Basis('Hyperbolic basis', [4, 4, 4], q=1.0)
        total = Basis('Total order', [4, 4, 4])
        np.testing.assert_almost_equal(hyper.cardinality, total.cardinality, decimal=7, err_msg = "Difference greated than imposed tolerance")

if __name__== '__main__':
    unittest.main()
