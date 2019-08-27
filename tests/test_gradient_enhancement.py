from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
def fun(x):
    return np.exp(2*x[0] + x[1])
def gradfun(x):
    return [2*np.exp(2*x[0] + x[1]), np.exp(2*x[0] + x[1])]
class TestC(TestCase):
    def test_over_and_under_sampling(self):
        x1 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        x2 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        totalorder = Basis('total-order')
        OBJECT = Poly(parameters=[x1, x2], basis=totalorder, method='least-squares-with-gradients',
            sampling_args={'mesh':'tensor-grid', 'sampling-ratio': 0.5,
                           'subsampling-algorithm': 'qr'})
        OBJECT.set_model(fun, gradfun)
        coefficients = OBJECT.get_coefficients()

        x1 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        x2 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        totalorder = Basis('total-order')
        OBJECT2 = Poly(parameters=[x1, x2], basis=totalorder, method='least-squares-with-gradients',
            sampling_args={'mesh':'tensor-grid', 'sampling-ratio': 1.0,
                           'subsampling-algorithm': 'qr'})
        OBJECT2.set_model(fun, gradfun)
        coefficients2 = OBJECT2.get_coefficients()

        x1 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        x2 = Parameter(distribution='Uniform', order=9, lower=-1., upper=1.)
        totalorder = Basis('total-order')
        OBJECT3 = Poly(parameters=[x1, x2], basis=totalorder, method='least-squares-with-gradients',
            sampling_args={'mesh':'monte-carlo', 'sampling-ratio': 1.5,
                           'subsampling-algorithm': 'svd'})
        OBJECT3.set_model(fun, gradfun)
        coefficients3 = OBJECT3.get_coefficients()

        np.testing.assert_array_almost_equal(coefficients, coefficients2, decimal=4)
        np.testing.assert_array_almost_equal(coefficients3, coefficients2, decimal=4)

        totalorder = Basis('total-order')
        sample_points = np.random.rand(1000,2) * 2.0  - 1.0
        sample_outputs = evaluate_model(sample_points, fun)
        sample_grads = evaluate_model_gradients(sample_points, gradfun, format='matrix')
        totalorder = Basis('total-order')
        OBJECT2 = Poly(parameters=[x1, x2], basis=totalorder, method='least-squares-with-gradients',
            sampling_args={'mesh':'user-defined', 'sample-points': sample_points, 'sample-outputs': sample_outputs, 'sample-gradients': sample_grads})
        OBJECT2.set_model(fun, gradfun)
        coefficients = OBJECT.get_coefficients()
if __name__== '__main__':
    unittest.main()