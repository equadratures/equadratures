from unittest import TestCase
import unittest
from equadratures.poly import Poly
from equadratures.sampling_methods.induced import Induced
from equadratures.parameter import Parameter
from equadratures.basis import Basis

import numpy as np


class TestSamplingGeneration(TestCase):
    def test_sampling(self):
        d = 4
        order = 5
        param = Parameter(distribution='uniform',
                          order=order,
                          lower=-1.0, upper=1.0)
        myparameters = [param for _ in range(d)]
        mybasis = Basis('total-order')
        mypoly = Poly(myparameters, mybasis,
                      method='least-squares',
                      sampling_args={'mesh': 'induced',
                                     'subsampling-algorithm': 'qr',
                                     'sampling-ratio': 1})

        assert mypoly._quadrature_points.shape == (mypoly.basis.cardinality, d)
        p, w = mypoly.get_points_and_weights()
        P = mypoly.get_poly(p)
        W = np.diag(np.sqrt(w))
        A = np.dot(W.T, P.T)
        G = np.dot(A.T, A)
        condition_number = np.linalg.cond(G)
        assert condition_number < 150

    def test_induced_sampling(self):
        """
        An integration test for the whole routine
        """
        dimension = 3
        parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
        basis = Basis("total-order", [3]*dimension)

        induced_sampling = Induced(parameters, basis)

        quadrature_points = induced_sampling.get_points()
        assert quadrature_points.shape == (induced_sampling.samples_number, 3)


if __name__ == '__main__':
    unittest.main()
