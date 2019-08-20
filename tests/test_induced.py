from unittest import TestCase
import unittest
from equadratures.poly import Poly
from equadratures.sampling_methods.induced import Induced
from equadratures.parameter import Parameter
from equadratures.basis import Basis

import numpy as np
# import time


class TestSamplingGeneration(TestCase):

    def test_sampling(self):
        d = 3
        order = 3
        param = Parameter(distribution='uniform', order=order, lower=-1.0, upper=1.0)
        myparameters = [param for _ in range(d)]
        mybasis2 = Basis('total-order')
        mypoly2 = Poly(myparameters, mybasis2, method='least-squares', sampling_args={'mesh':'induced', 'subsampling-algorithm':'qr', 'sampling-ratio':1.0})
        assert mypoly2._quadrature_points.shape == (mypoly2.basis.cardinality, d)

    def test_induced_jacobi_evaluation(self):
        dimension = 3
        parameters = [Parameter(1, "Uniform", upper=1, lower=-1)]*dimension
        basis = Basis("total-order")
        induced_sampling = Induced(parameters, basis)

        parameter = parameters[0]
        parameter.order = 3
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0, parameter)
        np.testing.assert_allclose(cdf_value, 0.5, atol=0.00001)
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 1, parameter)
        assert cdf_value == 1
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, -1, parameter)
        assert cdf_value == 0
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.6, parameter)
        np.testing.assert_allclose(cdf_value, 0.7462, atol=0.00005)
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.999, parameter)
        np.testing.assert_allclose(cdf_value, 0.99652, atol=0.000005)

if __name__ == '__main__':
    unittest.main()
