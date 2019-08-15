from unittest import TestCase
from equadratures.induced_sampling import InducedSampling
from equadratures.parameter import Parameter
from equadratures.basis import Basis

# import numpy as np
# from numpy.testing import assert_array_equal
import math


class TestSamplingGeneration(TestCase):

    # def test_samples(self):
    #     """
    #     test if the method returns a function object for sampling interface
    #     """
    #     dimension = 3
    #     sampling_ratio = 3
    #     parameters = [Parameter(1, "gaussian")]*dimension
    #     basis = Basis("total order", [5]*dimension)

    #     induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

    #     # Mock additive mixture sampling
    #     def func(array_):
    #         return np.array([1]*dimension, float)

    #     induced_sampling.additive_mixture_sampling = func
    #     quadrature_points = induced_sampling.samples()
    #     true_array = np.ones((dimension*sampling_ratio, dimension))
    #     assert_array_equal(quadrature_points, true_array)

    # def test_additive_mixture_sampling(self):
    #     """
    #     test if the method returns a function object for sampling interface
    #     """
    #     dimension = 3
    #     sampling_ratio = 3
    #     parameters = [Parameter(1, "gaussian")]*dimension
    #     basis = Basis("total order", [5]*dimension)

    #     induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

    #     # Mock multi_variate_sampling
    #     def func(sampled_cdf_values, index_set_used):
    #         return sampled_cdf_values, index_set_used

    #     induced_sampling.multi_variate_sampling = func
    #     _placeholder = np.ones((dimension, 1))
    #     cdf, index = induced_sampling.additive_mixture_sampling(_placeholder)
    #     assert type(cdf) == np.ndarray
    #     assert cdf.shape == (dimension, 1)
    #     assert cdf.dtype == 'float64'
    #     assert np.amax(cdf) < 1
    #     assert np.amin(cdf) > 0
    #     assert type(index) == np.ndarray
    #     assert index.shape == (3,)
    #     index_int = index.astype(int)
    #     assert np.all(np.isclose(index, index_int, 0.0001))
    #     # Check Total order
    #     assert np.sum(index) <= 5
    #     assert np.amax(index) <= 5
    #     assert np.amin(index) >= 0

    # def test_multi_variate_sampling(self):
    #     """
    #     test if the method returns a function object for sampling interface
    #     """
    #     dimension = 3
    #     sampling_ratio = 3
    #     parameters = [Parameter(1, "gaussian")]*dimension
    #     basis = Basis("total order", [5]*dimension)

    #     induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

    #     # Mock univariate sampling
    #     def func(_input):
    #         if isinstance(_input, tuple) and len(_input) == 3:

    #             parameter = _input[0]
    #             cdf = _input[1]
    #             order = _input[2]
    #             assert parameter.__class__ == Parameter
    #             assert cdf.shape == (1,)
    #             assert cdf < 1 and cdf > 0
    #             # Check integer property
    #             assert order - int(order) < 0.0001
    #             assert order <= 5 and order >= 0
    #             assert type(np.asscalar(order)) == float
    #             return 1
    #         else:
    #             return 0

    #     induced_sampling.univariate_sampling = func
    #     quadrature_points = induced_sampling.samples()
    #     true_array = np.ones((dimension*sampling_ratio, dimension))
    #     assert_array_equal(quadrature_points, true_array)

    def test_induced_jacobi_evaluation(self):

        dimension = 3
        sampling_ratio = 3
        parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
        basis = Basis("total order", [5]*dimension)

        induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

        parameter = parameters[0]
        parameter.order = 3
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0, parameter)
        assert math.isclose(cdf_value, 0.5, rel_tol=0.00001)
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 1, parameter)
        assert cdf_value == 1
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, -1, parameter)
        assert cdf_value == 0
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.6, parameter)
        print(cdf_value)
        assert math.isclose(cdf_value, 0.7462, abs_tol=0.00005)
        cdf_value = induced_sampling.induced_jacobi_evaluation(0, 0, 0.999, parameter)
        print(cdf_value)
        assert math.isclose(cdf_value, 0.99652, abs_tol=0.000005)

    # def test_inverse_induced_jacobi(self):
    #     """
    #     An integration test for the whole routine
    #     """
    #     dimension = 3
    #     sampling_ratio = 3
    #     parameters = [Parameter(55, "Uniform", upper=1, lower=-1)]*dimension
    #     basis = Basis("total order", [55]*dimension)

    #     induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

    #     parameter = parameters[0]
    #     parameter.order = 3
    #     sample = induced_sampling.inverse_induced_jacobi(0, 0, 0.74624, parameter)
    #     assert math.isclose(sample, 0.6, abs_tol=0.001)

    #     sample = induced_sampling.inverse_induced_jacobi(0, 0, 0.8082, parameter)
    #     assert math.isclose(sample, 0.9, abs_tol=0.001)

    #     sample = induced_sampling.inverse_induced_jacobi(0, 0, 0.99652094, parameter)
    #     assert math.isclose(sample, 0.999, abs_tol=0.0001)

    def test_induced_sampling(self):
        """
        An integration test for the whole routine
        """
        dimension = 3
        sampling_ratio = 10
        parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
        basis = Basis("total order", [3]*dimension)

        induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

        quadrature_points = induced_sampling.samples()

        assert quadrature_points.shape == (165, 3)
