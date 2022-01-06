from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from scipy.stats import linregress

class TestPolynet(TestCase):
    def test_nn_sl(self):
        np.random.seed(0)
        num_vars = 25
        num_ridges = 2
        w0w1 = np.random.randn(num_vars, 2)
        q = np.linalg.qr(w0w1)[0]
        w0 = q[:,0]
        w1 = q[:,1]

        def nice_func(x):
            u0 = np.dot(w0,x)
            u1 = np.dot(w1,x)
            return u0**2 + 2.0*u1**2 + 0.5*u1 - 0.5*u0 + 0.7

        N = 1000
        X = np.random.uniform(-1,1,(N,num_vars))
        Y = np.squeeze(evaluate_model(X, nice_func))
    
        net = Polynet(X,Y,num_ridges,max_iters=20000, learning_rate=1e-4, momentum_rate=.001, opt='adapt')
        net.fit()

        s,i,r,_,_ = linregress(net.evaluate_fit(X), Y)

        np.testing.assert_almost_equal(s, 1, decimal=3)
        np.testing.assert_almost_equal(i, 0, decimal=3)
        np.testing.assert_almost_equal(r, 1, decimal=3)

if __name__== '__main__':
    unittest.main()
