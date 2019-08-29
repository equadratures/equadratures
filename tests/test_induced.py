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
        param = Parameter(distribution='uniform', order=order, lower=-1.0, upper=1.0)
        myparameters = [param for _ in range(d)]
        mybasis2 = Basis('total-order')
        mypoly2 = Poly(myparameters, mybasis2, method='least-squares', sampling_args={'mesh':'induced', 'subsampling-algorithm':'qr', 'sampling-ratio':1})
        assert mypoly2._quadrature_points.shape == (mypoly2.basis.cardinality, d)
        p2, w2 = mypoly2.get_points_and_weights()
        P2 = mypoly2.get_poly(p2)
        W2 = np.diag(np.sqrt(w2))
        A2 = np.dot(W2.T, P2.T)
        G2 = np.dot(A2.T, A2)
        condition_number = np.linalg.cond(G2)
        assert condition_number < 150

    # def test_induced_jacobi_evaluation(self):
    #     dimension = 3
    #     parameters = [Parameter(1, "Uniform", upper=1, lower=-1)]*dimension
    #     basis = Basis("total-order")
    #     induced_sampling = Induced(parameters, basis)

    #     parameter = parameters[0]
    #     parameter.order = 3
    #     cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0, parameter)
    #     np.testing.assert_allclose(cdf_value, 0.5, atol=0.00001)
    #     cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 1, parameter)
    #     assert cdf_value == 1
    #     cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, -1, parameter)
    #     assert cdf_value == 0
    #     cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.6, parameter)
    #     np.testing.assert_allclose(cdf_value, 0.7462, atol=0.00005)
    #     cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.999, parameter)
    #     np.testing.assert_allclose(cdf_value, 0.99652, atol=0.000005)

    def test_induced_sampling(self):
        """
        An integration test for the whole routine
        """
        dimension = 3
        parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
        basis = Basis("total-order", [3]*dimension)

        induced_sampling = Induced(parameters, basis)

        quadrature_points = induced_sampling.get_points()
        assert quadrature_points.shape == (300, 3)


if __name__ == '__main__':
    unittest.main()
