from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

def model(x):
    return np.exp(10*x[0] + x[1])

def model2(x):
    return np.exp(x[0] + x[1])

def model1D(x):
    return np.exp(x[0])

class TestA(TestCase):

    def test_tensor_grid_coefficients(self):
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=30)
        basis = Basis('tensor-grid')
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration')
        pts, wts = poly.get_points_and_weights()
        model_evals = evaluate_model(pts, model)
        poly.set_model(model_evals)
        mean, variance = poly.get_mean_and_variance()
        np.testing.assert_almost_equal(mean, 1294.276442022, decimal=3, err_msg='Problem!')
        np.testing.assert_almost_equal(variance, 20320178.97284, decimal=3, err_msg='Problem!')
        model_evals2 = evaluate_model(pts, model2)
        poly.set_model(model_evals2)
        mean, variance = poly.get_mean_and_variance()
        np.testing.assert_almost_equal(mean, 1.381097845541819, decimal=3, err_msg='Problem!')
        np.testing.assert_almost_equal(variance, 1.3810978455418375, decimal=3, err_msg='Problem!')

    def test_sparse_grid_coefficients(self):
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=30)
        basis = Basis('sparse-grid', level=7, growth_rule='exponential')
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration')
        pts, wts = poly.get_points_and_weights()
        model_evals = evaluate_model(pts, model)
        poly.set_model(model_evals)
        mean, variance = poly.get_mean_and_variance()
        np.testing.assert_almost_equal(mean, 1294.276442022, decimal=3,err_msg='Problem!')
        np.testing.assert_almost_equal(variance, 20320178.96583, decimal=3, err_msg='Problem!')

    def test_univariate_quadrature_rules(self):

        # With end-points
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=20, endpoints=True)
        basis = Basis('univariate')
        poly = Poly([param], basis, method='numerical-integration')
        pts = poly.get_points()
        poly.set_model(model1D)
        np.testing.assert_array_almost_equal(float(pts[0]), -1.0, decimal=5, err_msg='Problem!')
        coefficients_lobatto = poly.get_coefficients()

        # Without end-points
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=20, endpoints=True)
        basis = Basis('univariate')
        poly = Poly([param], basis, method='numerical-integration')
        pts = poly.get_points()
        poly.set_model(model1D)
        coefficients_standard = poly.get_coefficients()

        np.testing.assert_array_almost_equal(coefficients_lobatto, coefficients_standard, decimal=3, err_msg='Problem!')

if __name__== '__main__':
    unittest.main()