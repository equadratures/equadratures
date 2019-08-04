from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

def rosenbrock_fun(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def exponential_fun(x):
    return np.exp(x[0] + x[1])

class TestA(TestCase):

    def test_tensor_grid_coefficients(self):

    def test_sparse_grid_coefficients(self):

    def tensor_grid_quadrature_rule(self):

    def sparse_grid_quadrature_rule(self):

if __name__== '__main__':
    unittest.main()



mu = 1
        sigma = 2
        variance = sigma**2
        x1 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=6)
        x2 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=6)
        parameters = [x1, x2]
        basis = Basis('Tensor grid')
        uqProblem = Poly(parameters, basis, 'numerical-integration')
        uqProblem.computeCoefficients(rosenbrock_fun)
        myStats = uqProblem.getStatistics()
        large_number = 1000000
        s = sigma * np.random.randn(large_number,2) + mu
        f = np.zeros((large_number,1))
        for i in range(0, large_number):
            f[i,0] = rosenbrock_fun([s[i,0], s[i,1]])
        np.testing.assert_almost_equal(myStats.mean * 1.0/10000.0, np.mean(f)* 1.0/10000.0, decimal=2, err_msg = "Difference greated than imposed tolerance")
        np.testing.assert_almost_equal(myStats.variance* 1.0/1000000000.0, np.var(f)* 1.0/1000000000.0, decimal=2, err_msg = "Difference greated than imposed tolerance")
