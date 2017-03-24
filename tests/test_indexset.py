from unittest import TestCase
import unittest
from effective_quadratures.indexset import IndexSet
import numpy as np

class TestIndexSet(TestCase):

    def test_indexsets(self):
        is1 = IndexSet('Tensor grid', [3,3,3])

        is2 = IndexSet('Sparse grid', level=3, growth_rule='linear', dimension=2)
        sparse_indices, sparse_coeffs, sparse_all_elements =  is2.getIndexSet()

        is3 = IndexSet('Sparse grid', level=2, growth_rule='exponential', dimension=3)
        sparse_indices, sparse_coeffs, sparse_all_elements =  is3.getIndexSet()

        is4 = IndexSet('Hyperbolic basis', [5,5], q=0.75)
        is4.getIndexSet()


    def test_sorting(self):
        is1 = IndexSet('Total order', [3,2])
        is1.prune(5)

    def test_euclidean(self):
        is2 = IndexSet('Euclidean degree', [7,7,7])


if __name__ == '__main__':
    unittest.main()
