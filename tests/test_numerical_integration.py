from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

def model(x):
    return np.exp(10*x[0] + x[1])

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

    def test_sparse_grid_weights(self):
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=30)
        basis = Basis('sparse-grid', level=2, growth_rule='linear')
        poly = Poly(parameters=[param, param, param], basis=basis, method='numerical-integration')
        true_weights = np.asarray([0.11846344, 0.08696371, 0.08696371 ,0.17392742 ,0.08696371, 0.08696371, \
            0.07716049 ,0.06944444, 0.13888889, 0.06944444 ,0.07716049 ,0.13888889, \
            0.27777778 ,0.12345679, 0.13888889, 0.07716049 ,0.12345679 ,0.06944444, \
            0.13888889 ,0.06944444, 0.07716049, 0.08696371 ,0.06944444 ,0.13888889, \
            0.06944444 ,0.06944444, 0.125     , 0.25       ,0.11111111 ,0.125, \
            0.06944444 ,0.16303629, 0.08696371, 0.13888889 ,0.25       ,0.16303629, \
            0.22222222 ,0.16303629, 0.25      , 0.13888889 ,0.08696371 ,0.11111111 ,\
            0.22222222 ,0.11111111, 0.16303629, 0.06944444 ,0.125      ,0.25, \
            0.11111111 ,0.125     , 0.06944444, 0.06944444 ,0.13888889 ,0.06944444, \
            0.08696371 ,0.23931434, 0.16303629, 0.16303629 ,0.32607258 ,0.16303629,\
            0.16303629 ,0.28444444, 0.11846344, 0.08696371 ,0.17392742 ,0.08696371,\
            0.07716049 ,0.13888889, 0.27777778, 0.12345679 ,0.13888889 ,0.07716049,\
            0.08696371 ,0.13888889, 0.25      , 0.16303629 ,0.22222222 ,0.16303629,\
            0.25       ,0.13888889, 0.08696371, 0.23931434 ,0.16303629 ,0.32607258,\
            0.16303629 ,0.28444444, 0.11846344, 0.17392742 ,0.27777778 ,0.23931434,\
            0.32607258 ,0.28444444, 0.44444444, 0.32607258 ,0.23931434 ,0.27777778,\
            0.17392742 ,0.11846344, 0.12345679, 0.22222222 ,0.44444444 ,0.19753086,\
            0.22222222 ,0.12345679, 0.16303629, 0.32607258 ,0.16303629 ,0.23931434,\
            0.08696371 ,0.13888889, 0.25      , 0.16303629 ,0.22222222 ,0.16303629,\
            0.25       ,0.13888889, 0.08696371, 0.07716049 ,0.13888889 ,0.27777778,\
            0.12345679 ,0.13888889, 0.07716049, 0.08696371 ,0.17392742 ,0.08696371,\
            0.11846344 ,0.12345679, 0.11111111, 0.22222222 ,0.11111111 ,0.12345679,\
            0.22222222 ,0.44444444, 0.19753086, 0.22222222 ,0.12345679 ,0.19753086,\
            0.11111111 ,0.22222222, 0.11111111, 0.12345679 ,0.16303629 ,0.16303629,\
            0.32607258 ,0.16303629, 0.16303629, 0.23931434 ,0.08696371 ,0.06944444,\
            0.13888889 ,0.06944444, 0.06944444, 0.125      ,0.25       ,0.11111111,\
            0.125      ,0.06944444, 0.16303629, 0.08696371 ,0.13888889 ,0.25,\
            0.16303629 ,0.22222222, 0.16303629, 0.25       ,0.13888889 ,0.08696371,\
            0.11111111 ,0.22222222, 0.11111111, 0.16303629 ,0.06944444 ,0.125,\
            0.25       ,0.11111111, 0.125     , 0.06944444 ,0.06944444 ,0.13888889,\
            0.06944444 ,0.08696371, 0.07716049, 0.06944444 ,0.13888889 ,0.06944444,\
            0.07716049 ,0.13888889, 0.27777778, 0.12345679 ,0.13888889 ,0.07716049,\
            0.12345679 ,0.06944444, 0.13888889, 0.06944444 ,0.07716049 ,0.08696371,\
            0.08696371 ,0.17392742, 0.08696371, 0.08696371 ,0.11846344])
        np.testing.assert_array_almost_equal(true_weights, poly.get_weights(), decimal=3, err_msg='Problem!')

    def test_univariate_quadrature_rules(self):
        # With end-points


        # Without end-points

if __name__== '__main__':
    unittest.main()