from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from copy import deepcopy

def model(x):
    return x[0]**2 + x[1]**3 - x[0]*x[1]**2

class TestF(TestCase):

    def test_tensor_grid_with_nans(self):
        # Without Nans!
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=4)
        basis = Basis('tensor-grid')
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration')
        pts, wts = poly.get_points_and_weights()
        model_evals = evaluate_model(pts, model)
        poly.set_model(model_evals)
        mean, variance = poly.get_mean_and_variance()
        # With Nans!
        model_evals_with_NaNs = deepcopy(model_evals)
        indices_to_set_to_NaN = np.asarray([1, 3, 9, 13])
        model_evals_with_NaNs[indices_to_set_to_NaN] = np.nan * indices_to_set_to_NaN.reshape(len(indices_to_set_to_NaN),1)
        basis2 = Basis('tensor-grid')
        poly2 = Poly(parameters=[param, param], basis=basis2, method='numerical-integration')
        poly2.set_model(model_evals_with_NaNs)
        mean, variance = poly2.get_mean_and_variance()
        mean_with_nans, variance_with_nans = poly2.get_mean_and_variance()
        # Verify!
        np.testing.assert_almost_equal(mean, mean_with_nans, decimal=7, err_msg='Problem!')
        np.testing.assert_almost_equal(variance, variance_with_nans, decimal=7, err_msg='Problem!')

if __name__== '__main__':
    unittest.main()