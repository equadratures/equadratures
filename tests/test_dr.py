from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


class TestDr(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.my_dr = dr()
        np.random.seed(0)
        W = np.random.randn(4, 2)
        cls.W = np.linalg.qr(W)[0]
        cls.X = np.random.uniform(low=-1.0, high=1.0, size=(1000, 4))

    @staticmethod
    def subspace_dist(U, V):
        return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)

    def fun(self, x):
        u = np.dot(x, self.W)
        return u[0] ** 2 - 0.1 * u[1] ** 2 - 2.0 * u[0] ** 2 * u[1] ** 2 + 2.0 * u[0] * u[1]

    def test_standard(self):
        X_nonstan = np.random.uniform(low=-2.0, high=3.0, size=(1000, 4))
        bounds = np.ones((4, 2)) * np.array([-2, 3])
        X_stan = self.my_dr.standard(bounds, X=X_nonstan)
        assert np.max(X_stan) < 1.0
        assert np.min(X_stan) > -1.0

    def test_AS(self):
        p = 4
        my_basis = Basis('tensor grid', [p, p, p, p])
        my_params = [Parameter(p, distribution='uniform', lower=-1.0, upper=1.0) for _ in range(4)]
        my_poly = Polyint(my_params, my_basis)
        my_poly.computeCoefficients(self.fun)

        evecs = self.my_dr.computeActiveSubspaces(samples=self.X, poly=my_poly)[1]
        np.testing.assert_almost_equal(subspace_dist(evecs[:,:2], self.W), 0.0, decimal=5)

    def test_VP(self):
        np.random.seed(0)
        W = np.random.randn(4, 2)
        W = np.linalg.qr(W)[0]
        y = utils.evalfunction(self.X, self.fun)
        evecs = self.my_dr.variable_projection(2, 4, tol=1e-7, X=self.X, f=y)[0]
        np.testing.assert_almost_equal(subspace_dist(evecs[:,:2], self.W), 0.0, decimal=5)

    def test_linear(self):
        w = np.random.randn(4, 1)
        w /= np.linalg.norm(w)
        y = np.dot(self.X, w)

        u, _ = self.my_dr.linearModel(training_input=self.X, training_output=y)
        u = np.squeeze(u / np.linalg.norm(u))

        np.testing.assert_almost_equal(np.dot(u,w), 1.0, decimal=5)

    def test_vecAS(self):

        def fun0(x):
            u = np.dot(x, self.W)
            return u[0] ** 2 - 0.1 * u[1] ** 2 - 2.0 * u[0] ** 2 * u[1] ** 2

        def fun1(x):
            u = np.dot(x, self.W)
            return 2.0 * u[0] * u[1]

        p = 4
        my_basis = Basis('tensor grid', [p, p, p, p])
        my_params = [Parameter(p, distribution='uniform', lower=-1.0, upper=1.0) for _ in range(4)]
        my_poly0 = Polyint(my_params, my_basis)
        my_poly0.computeCoefficients(fun0)
        my_poly1 = Polyint(my_params, my_basis)
        my_poly1.computeCoefficients(fun1)

        evecs = self.my_dr.vector_AS([my_poly0, my_poly1])[1]
        np.testing.assert_almost_equal(subspace_dist(evecs[:, :2], self.W), 0.0, decimal=5)

