from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from scipy.stats import skew, kurtosis

def rosenbrock_fun(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

class TestD(TestCase):

    def test_sparse_grid_moments(self):
        mu = 1
        sigma = 2
        variance = sigma**2
        x1 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=6)
        x2 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=6)
        parameters = [x1, x2]
        parameters = [x1, x2]
        basis = Basis('sparse-grid', level=5, growth_rule='linear')
        uqProblem = Poly(parameters, basis, method='numerical-integration')
        uqProblem.set_model(rosenbrock_fun)
        mean, variance = uqProblem.get_mean_and_variance()
        skewness, kurtosis = uqProblem.get_skewness_and_kurtosis()

        large_number = 2000000
        s = sigma * np.random.randn(large_number,2) + mu
        model_evals = evaluate_model(s, rosenbrock_fun)
        mean_mc = np.mean(model_evals)
        variance_mc = np.var(model_evals)
        skewness_mc = skew(model_evals)

        np.testing.assert_almost_equal()




if __name__== '__main__':
    unittest.main()