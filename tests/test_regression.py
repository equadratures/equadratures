from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import scipy.stats as st

class TestC(TestCase):

    def test_quadratic(self):
        dimensions = 1
        M = 12
        param = Parameter(distribution='Uniform', lower=0, upper=1., order=2)
        myParameters = [param for i in range(dimensions)] # one-line for loop for parameters
        x_train = np.asarray([0.0,0.0714,0.1429,0.2857,0.3571,0.4286,0.5714,0.6429,0.7143,0.7857,0.9286,1.0000])
        y_train = np.asarray([6.8053,-1.5184,1.6416,6.3543,14.3442,16.4426,18.1953,28.9913,27.2246,40.3759,55.3726,72.0])
        x_train = np.reshape(x_train, (M, 1))
        y_train = np.reshape(y_train, (M, 1))

        myBasis = Basis('univariate')
        poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':x_train, 'sample-outputs':y_train})
        poly.set_model()
        coefficients = poly.get_coefficients().reshape(3, )
        true_coefficients = np.asarray([22.47470337, 17.50891379, 4.97964868])
        np.testing.assert_array_almost_equal(coefficients, true_coefficients, decimal=4, err_msg='Problem!')

    def test_polyvar(self):
        N = 10
        our_function = lambda x:  0.3*x**4 -1.6*x**3 +0.6*x**2 +2.4*x - 0.5
        n = 4 # degree of polynomial
        dimensions = 1
        
        x_train = np.asarray([-0.64632395, -0.18877934, -0.01358774, 0.45976263, 0.46337817, 0.66119015, 0.68384465, 0.9602067, 0.96965, 0.98821018])
        y_train = np.asarray([-1.08314486, -0.72882292, -0.93002666, 0.42621992, 0.73289331, 0.98535082, 0.85108678, 1.15953546, 1.41458158, 0.97508757])
        x_test = np.linspace(-1.5, 1.5, 5)
        
        param = Parameter(distribution='Uniform', lower=-1, upper=1, order=n)
        myBasis = Basis('Univariate')
        poly = Poly(param, myBasis, method='least-squares', sampling_args={'sample-points':x_train.reshape(-1,1), 'sample-outputs':y_train.reshape(-1,1)} )
        poly.set_model()
        
        testvar = poly.get_polyvar(x_test.reshape(-1,1))
        std = np.sqrt(testvar)
        true_std = [12.83481772],[0.41590942],[0.12530528],[0.1271031],[3.61383742]

        np.testing.assert_array_almost_equal(std, true_std, decimal=7, err_msg='Problem!')

if __name__== '__main__':
    unittest.main()
