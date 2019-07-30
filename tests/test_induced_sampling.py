from unittest import TestCase
from equadratures.induced_sampling import InducedSampling
from equadratures.parameter import Parameter
from equadratures.basis import Basis

import numpy as np
from numpy.testing import assert_array_equal


class TestSamplingGeneration(TestCase):

    def test_samples(self):
        """
        test if the method returns a function object for sampling interface
        """
        dimension = 3
        sampling_ratio = 3
        parameters = [Parameter(1, "gaussian")]*dimension
        basis = Basis("total order", [5]*dimension)

        induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

        # Mock additive mixture sampling
        def func(array_):
            return np.array([1]*dimension, float)

        induced_sampling.additive_mixture_sampling = func
        quadrature_points = induced_sampling.samples()
        true_array = np.ones((dimension*sampling_ratio, dimension))
        assert_array_equal(quadrature_points, true_array)

    def test_additive_mixture_sampling(self):
        """
        test if the method returns a function object for sampling interface
        """
        dimension = 3
        sampling_ratio = 3
        parameters = [Parameter(1, "gaussian")]*dimension
        basis = Basis("total order", [5]*dimension)

        induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

        # Mock multi_variate_sampling
        def func(sampled_cdf_values, index_set_used):
            return sampled_cdf_values, index_set_used

        induced_sampling.multi_variate_sampling = func
        _placeholder = np.ones((dimension, 1))
        cdf, index = induced_sampling.additive_mixture_sampling(_placeholder)
        assert type(cdf) == np.ndarray
        assert cdf.shape == (dimension, 1)
        assert cdf.dtype == 'float64'
        assert np.amax(cdf) < 1
        assert np.amin(cdf) > 0
        assert type(index) == np.ndarray
        assert index.shape == (3,)
        index_int = index.astype(int)
        assert np.all(np.isclose(index, index_int, 0.0001))
        # Check Total order
        assert np.sum(index) <= 5
        assert np.amax(index) <= 5
        assert np.amin(index) >= 0

    def test_multi_variate_sampling(self):
        """
        test if the method returns a function object for sampling interface
        """
        dimension = 3
        sampling_ratio = 3
        parameters = [Parameter(1, "gaussian")]*dimension
        basis = Basis("total order", [5]*dimension)

        induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

        # Mock univariate sampling
        def func(_input):
            if isinstance(_input, tuple) and len(_input) == 3:

                parameter = _input[0]
                cdf = _input[1]
                order = _input[2]
                assert parameter.__class__ == Parameter
                assert cdf.shape == (1,)
                assert cdf < 1 and cdf > 0
                # Check integer property
                assert order - int(order) < 0.0001
                assert order <= 5 and order >= 0
                assert type(np.asscalar(order)) == float
                return 1
            else:
                return 0

        induced_sampling.univariate_sampling = func
        quadrature_points = induced_sampling.samples()
        true_array = np.ones((dimension*sampling_ratio, dimension))
        assert_array_equal(quadrature_points, true_array)
