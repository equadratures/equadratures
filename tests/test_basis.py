from unittest import TestCase
import unittest
from equadratures import *

class TestBasis(TestCase):

    def test_declaration(self):
        is1 = Basis('Tensor grid', [3,3,3])

        is2 = Basis('Sparse grid', level=3, growth_rule='linear')
        is2.setOrders([3, 3, 3])
        sparse_indices, sparse_coeffs, sparse_all_elements =  is2.getBasis()

        is3 = Basis('Sparse grid', level=2, growth_rule='exponential')
        is3.setOrders([5, 5])
        sparse_indices, sparse_coeffs, sparse_all_elements =  is3.getBasis()

        is4 = Basis('Hyperbolic basis', [5,5], q=0.75)
        is4.getBasis()

    def test_sorting(self):
        is1 = Basis('Total order', [3,2])
        is1.prune(5)

    def test_euclidean(self):
        is2 = Basis('Euclidean degree', [7,7,7])

    def test_plot(self):
        is1 = Basis('Total order', [3,2])
        scatterplot2(is1.elements[:,0], is1.elements[:,1], '$i_1$', '$i_2$', filename='basis.eps')


if __name__ == '__main__':
    unittest.main()
