from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

def phi(x):
    # first normalized Legendre polynomial...?
    return np.sqrt(3) * x


def fun(X):
    x = phi(X)
    return 0.1 + 0.2 * x[0] + 0.3 * x[1] * x[2] + 0.4 * x[1] * x[2] * x[3] + 0.5 * x[0] * x[1] * x[2] * x[3]

class TestStats(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        p_order = 4
        my_basis = Basis('total order', orders=[p_order, p_order, p_order, p_order])
        my_params = [Parameter(order=p_order, distribution='uniform', lower=-1, upper=1) for _ in range(4)]

        X = np.random.uniform(-1, 1, (10000, 4))
        my_poly = Polyreg(my_params, my_basis, training_inputs=X, fun=fun)
        cls.my_stats = my_poly.getStatistics()

    def test_mean(self):
        np.testing.assert_almost_equal(self.my_stats.mean, 0.1, decimal=5)

    def test_var(self):
        analytical_variance = 0.2**2 + 0.3**2 + 0.4**2 + 0.5**2 # == 0.54
        np.testing.assert_almost_equal(self.my_stats.variance, 0.54, decimal=5)

    def test_skew(self):

        # Can verify using the following:
        # p = sp.Poly((0.2 * x0 + 0.3 * x1 * x2 + 0.4 * x1 * x2 * x3 + 0.5 * x0 * x1 * x2 * x3) ** 3)
        # monoms = p.monoms()
        # coeffs = p.coeffs()
        # integral_sum = 0
        # skew_indices = {}
        # for i in range(len(monoms)):
        #     if 1 not in monoms[i] and 3 not in monoms[i]:
        #         integral_sum += coeffs[i]
        #         involved_variables = tuple(np.nonzero(monoms[i])[0])
        #         if involved_variables not in skew_indices.keys():
        #             skew_indices[involved_variables] = coeffs[i]
        #         else:
        #             skew_indices[involved_variables] += coeffs[i]
        # for i in skew_indices.keys():
        #     skew_indices[i] /= integral_sum

        np.testing.assert_almost_equal(self.my_stats.skewness, 0.60481, decimal=5)

    def test_kurt(self):

        # Can verify using the following:
        # x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
        # p = sp.Poly((0.2 * x0 + 0.3 * x1 * x2 + 0.4 * x1 * x2 * x3 + 0.5 * x0 * x1 * x2 * x3) ** 4)
        # monoms = p.monoms()
        # coeffs = p.coeffs()
        # integral_sum = 0
        # kurt_indices = {}
        # for i in range(len(monoms)):
        #     if 1 not in monoms[i] and 3 not in monoms[i]:
        #         term = [1.8 if monoms[i][j] == 4 else 1.0 for j in range(4)]
        #         integral_sum += coeffs[i] * np.prod(term) / 0.54 ** 2
        #         involved_variables = tuple(np.nonzero(monoms[i])[0])
        #         if involved_variables not in kurt_indices.keys():
        #             kurt_indices[involved_variables] = coeffs[i] * np.prod(term) / 0.54 ** 2
        #         else:
        #             kurt_indices[involved_variables] += coeffs[i] * np.prod(term) / 0.54 ** 2
        # for i in kurt_indices.keys():
        #     kurt_indices[i] /= integral_sum

        np.testing.assert_almost_equal(self.my_stats.kurtosis, 10.69801, decimal=5)

    def test_skew_indices(self):

        condskew4 = self.my_stats.getCondSkewness(4)
        np.testing.assert_almost_equal(condskew4[(0,1,2,3)], 1.0, decimal=5)

    def test_kurt_indices(self):
        condkurt1 = self.my_stats.getCondKurtosis(1)
        condkurt2 = self.my_stats.getCondKurtosis(2)
        condkurt3 = self.my_stats.getCondKurtosis(3)
        condkurt4 = self.my_stats.getCondKurtosis(4)

        np.testing.assert_almost_equal(condkurt1[(0, )], 0.0009232, decimal=5)
        np.testing.assert_almost_equal(condkurt2[(1, 2)], 0.008413, decimal=5)
        np.testing.assert_almost_equal(condkurt3[(0,1,2)], 0.006924, decimal=5)
        np.testing.assert_almost_equal(condkurt3[(1,2,3)], 0.137596, decimal=5)
        np.testing.assert_almost_equal(condkurt4[(0,1,2,3)], 0.846144, decimal=5)
