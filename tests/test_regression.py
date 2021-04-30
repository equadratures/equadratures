from unittest import TestCase
import unittest
from equadratures import *
from equadratures import datasets

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

    def test_robust(self):
        """
        Tests robust regression (huber and least-absolute-residual), with osqp and scipy backends. 
        """
        methods = ['huber','least-absolute-residual']
        opts    = ['osqp','scipy'] 
        f = lambda x:  (-0.3*x**4 -3*x**3 +0.6*x**2 +2.4*x - 0.5)

        N = 50 # number of training points (note, some will be removed below)
        n = 4 # degree of polynomial
        state = 15 # random seed
        
        # Add some noise
        noise_var = 0.1
        x = np.sort(np.random.RandomState(state).uniform(-1,1,N))
        y = f(x) + np.random.RandomState(state).normal(0,noise_var,size=N).T
        
        # delete training points between 0 < x < 0.3
        pos = ((x>0)*(x<0.3)).nonzero()[0]
        x = np.delete(x,pos)
        y = np.delete(y,pos)

        # Add some outliers
        randrange = range(10,17)
        y[randrange] = y[randrange]+np.random.RandomState(1).normal(0,4**2,len(randrange))
        
        # Test data
        x = x.reshape(-1,1)
        xtest = np.linspace(-1,1,100).reshape(-1,1)
        ytest = f(xtest)

        # param and basis
        param = Parameter(distribution='uniform', lower=-1, upper=1, order=n)
        basis = Basis('univariate')

        # Test Poly regressions
        for method in methods:
            for opt in opts:
                if method != 'huber' and opt != 'scipy': # TODO - remove this if statement once scipy huber regression implemented
                    poly = Poly(parameters=param, basis=basis, method=method,
                            sampling_args= {'mesh': 'user-defined', 'sample-points':x.reshape(-1,1), 'sample-outputs': y.reshape(-1,1)},
                            solver_args={'M':0.2**2,'verbose':False,'optimiser':opt})
                    poly.set_model()
                    _,r2 = poly.get_polyscore(X_test=xtest,y_test=ytest)
                    self.assertTrue(r2 > 0.997,msg='Poly method = %a, optimiser = %a' %(method,opt))

    def test_ElasticNet_linear(self):
        """ 
        Tests elastic-net regularisation on linear (1st order) synthetic data with irrelevent features. (using cvxpy LP solve here).
        """
        # Load linear dataset with n_observations=500,n_dim=10,bias=0.5,n_relevent=2,noise=0.2,train/test split = 0.8
        data = np.load('./tests/test_data/linear_data.npz')
        X_train = data['X_train']; y_train = data['y_train']; X_test = data['X_test']; y_test = data['y_test']

        # Define param and basis
        s = Parameter(distribution='uniform', lower=-1, upper=1, order=1,endpoints='both')
        param = [s for _ in range(X_train.shape[1])]
        basis = Basis('total-order') 
    
        # Fit Poly with OLS and Elastic Net (but with lambda=0 so effectively OLS) and check r2 scores match
        poly_OLS = Poly(parameters=param, basis=basis, method='least-squares', 
                sampling_args= {'mesh': 'user-defined', 'sample-points':X_train, 'sample-outputs': y_train.reshape(-1,1)})
        poly_OLS.set_model()
        _,r2_OLS = poly_OLS.get_polyscore(X_test=X_test,y_test=y_test)
    
        poly_EN = poly = Poly(parameters=param, basis=basis, method='elastic-net', 
                  sampling_args= {'mesh': 'user-defined', 'sample-points':X_train, 'sample-outputs': y_train.reshape(-1,1)},
                  solver_args={'path':False,'lambda':0.0,'alpha':0.5})
        poly_EN.set_model()
        _,r2_EN = poly_EN.get_polyscore(X_test=X_test,y_test=y_test)
    
        np.testing.assert_array_almost_equal(r2_OLS,r2_EN, decimal=4, err_msg='Problem!')

        # Now fit Poly with LASSO (alpha = 1.0) and check r2 improved (it should because irrelevent features + noise)
        poly_LASSO = Poly(parameters=param, basis=basis, method='elastic-net', 
                  sampling_args= {'mesh': 'user-defined', 'sample-points':X_train, 'sample-outputs': y_train.reshape(-1,1)},
                  solver_args={'path':False,'lambda':0.015,'alpha':1.0})
        poly_LASSO.set_model()
        _,r2_LASSO = poly_LASSO.get_polyscore(X_test=X_test,y_test=y_test)
        self.assertTrue(r2_LASSO > r2_OLS)

        # Finally, check LASSO has shrunk irrelevent Poly coefficients
        coeffs = poly_LASSO.get_coefficients().squeeze()
        ideal_coeffs = 3 #As tensor-grid, order=1, relevent_dims=2
        idx = np.abs(coeffs).argsort()[::-1]
        irrelevent_coeffs = np.sum(np.abs(coeffs[idx[ideal_coeffs:]]))/np.sum(np.abs(coeffs))
        self.assertTrue(irrelevent_coeffs < 1e-5,msg='irrelevent_coeffs = %.2e' %irrelevent_coeffs)

    def test_ElasticNet_friedman(self):
        """ 
        Tests elastic-net regularisation on quadratic (2nd order) synthetic data with irrelevent features. (using coord descent here).
        """
        # Generate friedman dataset with n_observations=200,n_dim=10,noise=0.2,normalise=False,train/test split of 0.8
        data = np.load('./tests/test_data/friedman_data.npz')
        X_train = data['X_train']; y_train = data['y_train']; X_test = data['X_test']; y_test = data['y_test']

        # Define param and basis
        s = Parameter(distribution='uniform', lower=-1, upper=1, order=2,endpoints='both')
        param = [s for _ in range(X_train.shape[1])]
        basis = Basis('total-order') 

        # Fit OLS poly
        poly_OLS = Poly(parameters=param, basis=basis, method='least-squares', 
                sampling_args= {'mesh': 'user-defined', 'sample-points':X_train, 'sample-outputs': y_train.reshape(-1,1)})
        poly_OLS.set_model()
        _,r2_OLS = poly_OLS.get_polyscore(X_test=X_test,y_test=y_test)

        # Fit Poly with LASSO (alpha = 1.0) and check r2 improved
        print('running elastic net')
        poly_LASSO = Poly(parameters=param, basis=basis, method='elastic-net', 
                  sampling_args= {'mesh': 'user-defined', 'sample-points':X_train, 'sample-outputs': y_train.reshape(-1,1)},
                  solver_args={'path':True,'alpha':1.0})

        poly_LASSO.set_model()
        _,r2_LASSO = poly_LASSO.get_polyscore(X_test=X_test,y_test=y_test)
        self.assertTrue(r2_LASSO > r2_OLS)
        print(r2_LASSO,r2_OLS)
   
        # Finally, check LASSO has shrunk L1norm of coefficients
        coeffs_OLS   = poly_OLS.get_coefficients().squeeze()
        coeffs_LASSO = poly_LASSO.get_coefficients().squeeze()
        l1_OLS   = np.linalg.norm(coeffs_OLS,ord=1)
        l1_LASSO = np.linalg.norm(coeffs_LASSO,ord=1)
        self.assertTrue(l1_LASSO < l1_OLS,msg='l1_LASSO = %.2e, l1_OLS = %.2e' %(l1_LASSO,l1_OLS))

    def test_polyuq_empirical(self):
        """ 
        Tests the get poly eq routine when no variance data is given, i.e. when estimating the 
        empirical variance from training data.
        """
#        # Generate data
        n = 1
       # Load linear dataset with n_observations=500,n_dim=10,bias=0.5,n_relevent=2,noise=0.2,train/test split = 0.8
        data = np.load('./tests/test_data/linear_data.npz')
        X_train = data['X_train']; y_train = data['y_train']
        N,dim = X_train.shape

        # Fit poly and approx its variance
        param = Parameter(distribution='Uniform', lower=-1, upper=1, order=n)
        myParameters = [param for i in range(dim)] # one-line for loop for parameters
        myBasis = Basis('total-order')
        poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':X_train, 'sample-outputs':y_train.reshape(-1,1)} )
        poly.set_model()
        y_pred, y_std = poly.get_polyfit(X_train,uq=True)
        np.testing.assert_array_almost_equal(y_std.mean(), 0.67484, decimal=5, err_msg='Problem!')

    def test_polyuq_prescribed(self):
        """
        Tests the get poly uq routine when variance data is given in sampling_args. 
        """
        # Generate data
        dim = 1
        n = 5
        N = 100
        our_function = lambda x:  0.3*x**4 -1.6*x**3 +0.6*x**2 +2.4*x - 0.5
        X = np.linspace(-1,1,N)
        y = our_function(X)

        # Array of prescribed variances at each training data point
#        y_var = state.uniform(0.05,0.2,N_train)**2
        y_var = 0.1*our_function(X)*X
        
        # Fit poly with prescribed variances
        param = Parameter(distribution='Uniform', lower=-1, upper=1, order=n)
        myParameters = [param for i in range(dim)] # one-line for loop for parameters
        myBasis = Basis('univariate')
        poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':X.reshape(-1,1), 
                                                                                  'sample-outputs':y.reshape(-1,1), 
                                                                                  'sample-output-variances':y_var} )
        poly.set_model()
        y_pred, y_std = poly.get_polyfit(X,uq=True)

        np.testing.assert_array_almost_equal(y_std.mean(), 0.64015, decimal=5, err_msg='Problem!')

if __name__== '__main__':
    unittest.main()

