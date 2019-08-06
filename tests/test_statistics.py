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
        del variance
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

        np.testing.assert_array_less(np.abs(mean-mean_mc)/mean * 100.0,  1.0)
        np.testing.assert_array_less(np.abs(variance-variance_mc)/variance * 100.0,  5.0)
        np.testing.assert_array_less(np.abs(skewness-skewness_mc)/skewness * 100.0,  5.0)

    def test_parameter_mc_least_squares_moments(self):
        x1 = Parameter(distribution='uniform',lower =0.0, upper=1.0, order=7)
        x2 = Parameter(distribution='Beta',lower =0.0, upper=1.0, shape_parameter_A = 1.6, shape_parameter_B=3.2 , order=7)
        myparameters = [x1, x2]
        mybasis = Basis('tensor-grid')
        mypoly = Poly(myparameters, mybasis, method='numerical-integration')
        mypoly.set_model(rosenbrock_fun)
        mean, variance = mypoly.get_mean_and_variance()
        skewness, kurtosis = mypoly.get_skewness_and_kurtosis()
        large_number = 3000000
        s = np.random.rand(large_number, 2)
        x1_samples = x1.get_icdf(s[:,0])
        x2_samples = x2.get_icdf(s[:,1])
        s = np.vstack([x1_samples, x2_samples]).T
        model_evals = evaluate_model(s, rosenbrock_fun)
        mean_mc = np.mean(model_evals)
        variance_mc = np.var(model_evals)
        skewness_mc = skew(model_evals)
        np.testing.assert_array_less(np.abs(mean-mean_mc)/mean * 100.0,  1.0)
        np.testing.assert_array_less(np.abs(variance-variance_mc)/variance * 100.0,  5.0)
        np.testing.assert_array_less(np.abs(skewness-skewness_mc)/skewness * 100.0,  5.0)


if __name__== '__main__':
    unittest.main()