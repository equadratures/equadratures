from unittest import TestCase
import unittest
from effective_quadratures.indexset import IndexSet
import numpy as np

class TestIndexSet(TestCase):
    
    def test_indexsets(self):
        is1 = IndexSet('Tensor grid', [3,3,3])
        print is1.getIndexSet()
        print '\n'

        is2 = IndexSet('Sparse grid', level=3, growth_rule='linear', dimension=2)
        sparse_indices, sparse_coeffs, sparse_all_elements =  is2.getIndexSet()
        print sparse_indices
        print '\n'

        is3 = IndexSet('Sparse grid', level=2, growth_rule='exponential', dimension=3)
        sparse_indices, sparse_coeffs, sparse_all_elements =  is3.getIndexSet()
        print sparse_indices
        print '\n'

        is4 = IndexSet('Hyperbolic basis', [5,5], q=0.75)
        print is4.getIndexSet()
        print '\n'

if __name__ == '__main__':
    unittest.main()