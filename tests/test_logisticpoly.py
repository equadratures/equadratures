from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestLogisticPoly(TestCase):
    def test_simple_poly(self):
        true_M = np.array([[-8.72517794e-01,  4.47658531e-01],
                           [-2.18844832e-01,  1.12058206e-04],
                           [-4.36829073e-01, -8.94204578e-01]])

        n = 2

        def true_poly(x):
            return (1.0 + 2.0 * x[0] + 3.0 * x[1]**2)

        rng = np.random.default_rng(42)
        M0 = np.linalg.qr(rng.normal(size=(3,2)))[0]

        X = rng.uniform(-1, 1, (1000,3))
        X_test = rng.uniform(-1,1, (100,3))
        my_params = [Parameter(order=2, distribution='uniform', lower=-1, upper=1) for _ in range(n)]
        my_basis = Basis('tensor-grid')
        pMx = np.apply_along_axis(true_poly, 1, X @ true_M)
        true_ortho_poly = Poly(parameters=my_params, basis=my_basis, method='least-squares'
                                  , sampling_args={'mesh':'user-defined',
                                                   'sample-points': X@true_M,
                                                   'sample-outputs': pMx})
        true_ortho_poly.set_model()
        true_c = true_ortho_poly.coefficients

        dummy_poly = Poly(parameters=my_params, basis=my_basis, method='least-squares')
        def sigmoid(U):
            return 1.0 / (1.0 + np.exp(-U))
        def p(X, M, c):
            dummy_poly.coefficients = c
            return dummy_poly.get_polyfit(X @ M).reshape(-1)
        def phi(X, M, c):
            pW = p(X,M,c)
            return sigmoid(pW)

        f = np.round(phi(X, true_M, true_c))
        f_test = np.round(phi(X_test, true_M, true_c))

        my_logistic_poly = LogisticPoly(n=2, cauchy_tol=1e-5, verbosity=0
                                           , order=2, max_M_iters=50, C=0.001, restarts=1,
                                        M_init=M0)
        my_logistic_poly.fit(X, f)
        prediction = my_logistic_poly.predict(X_test)
        error_rate = np.sum(np.abs(np.round(prediction) - f_test)) / f_test.shape[0]
        np.testing.assert_array_less(error_rate, 0.05, 'Classification error larger than 0.05.')
