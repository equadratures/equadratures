from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestBasis(TestCase):

    def test_tensor(self):
        tensor = Basis('tensor-grid', [4, 4, 4])
        np.testing.assert_almost_equal(tensor.cardinality, 125, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_sparse(self):
        sparse = Basis('sparse-grid', orders=[3,3,3], level=2, growth_rule='linear')
        a, b, c = sparse.get_basis()
        np.testing.assert_almost_equal(31, len(a), decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_euclidean(self):
        euclid = Basis('euclidean-degree', [4, 4, 4])
        np.testing.assert_almost_equal(euclid.cardinality, 54, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_hyperbolic(self):
        hyper = Basis('hyperbolic-basis', [4, 4, 4], q=0.5)
        np.testing.assert_almost_equal(hyper.cardinality, 16, decimal=7, err_msg = "Difference greated than imposed tolerance")

    def test_total(self):
        hyper = Basis('hyperbolic-basis', [4, 4, 4], q=1.0)
        total = Basis('total-order', [4, 4, 4])
        np.testing.assert_almost_equal(hyper.cardinality, total.cardinality, decimal=7, err_msg = "Difference greated than imposed tolerance")

if __name__== '__main__':
    unittest.main()
