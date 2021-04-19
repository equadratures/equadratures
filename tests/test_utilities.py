"""
This script tests utility functions such as those in datasets.py and scalers.py. 

List of functions/methods/classes which are already covered in other tests:
    - gen_friedman(): In test_regression.py 
    - train_test_split(): In test_regression.py
"""
from unittest import TestCase
import unittest
from equadratures import datasets
import numpy as np
from copy import deepcopy

class Test_Utilities(TestCase):

    def test_scalers(self):
        """
        Tests the scaler classes.
        """
        from equadratures.scalers import scaler_minmax, scaler_meanvar
    
        # Generate X data
        Xorig,_ = datasets.gen_linear(n_observations=500,n_dim=10,random_seed=42)
        Xorig *=10
    
        # Min/max scaler
        scaler = scaler_minmax()
        scaler.fit(Xorig)
        X = scaler.transform(Xorig)
        np.testing.assert_array_almost_equal(X.max(),  1.0, decimal=5, err_msg='Problem!')
        np.testing.assert_array_almost_equal(X.min(), -1.0, decimal=5, err_msg='Problem!')
        X = scaler.untransform(X)
        np.testing.assert_array_almost_equal(X, Xorig, decimal=5, err_msg='Problem!')

        # Mean/variance scaler
        scaler = scaler_meanvar()
        scaler.fit(Xorig)
        X = scaler.transform(Xorig)
        np.testing.assert_array_almost_equal(X.mean(), 0.0, decimal=5, err_msg='Problem!')
        np.testing.assert_array_almost_equal(X.var(),  1.0, decimal=5, err_msg='Problem!')
        X = scaler.untransform(X)
        np.testing.assert_array_almost_equal(X, Xorig, decimal=5, err_msg='Problem!')

    def test_scores(self):
        """
        Test the datasets.score() function.
        """
        x = np.array([0.1,0.05,0.4,0.2,0.31,0.34,0.5,0.6]).reshape(-1,1)
        y = deepcopy(x)
        y[0] += 0.02
        y[2] -= 0.01
        y[5] += 0.04
        score = datasets.score(x,y,'r2')
        np.testing.assert_array_almost_equal(score, 0.99301, decimal=5, err_msg='Problem!')
        score = datasets.score(x,y,'adjusted_r2',X=x)
        np.testing.assert_array_almost_equal(score, 0.99185, decimal=5, err_msg='Problem!')
        score = datasets.score(x,y,'mae')
        np.testing.assert_array_almost_equal(score, 0.00875, decimal=5, err_msg='Problem!')
        score = datasets.score(x,y,'normalised_mae')
        np.testing.assert_array_almost_equal(score, 0.04921, decimal=5, err_msg='Problem!')
        score = datasets.score(x,y,'rmse')
        np.testing.assert_array_almost_equal(score, 0.0162, decimal=5, err_msg='Problem!')

    def test_dataloader(self):
        """
        Test the load_eq_dataset() function.

        NOTE: This adds an external dependency to the EQ tests, since it relies on the data-sets repo.
        """
        dataset = 'naca0012'
        data = datasets.load_eq_dataset(dataset)
        self.assertIsInstance(data,np.lib.npyio.NpzFile)

    def test_friedman(self):
        """
        Test the gen_friedman() sythetic dataset generator. gen_linear() is tested implicitly by test_scalers(). 
        """
        N = 200
        d = 6
        X,y = datasets.gen_friedman(n_observations=N, n_dim=d, noise=0.0, normalise=False)
        np.testing.assert_equal(X.shape,np.array([N,d]))
        np.testing.assert_equal(y.shape,np.array([N,]))
        ytest = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] 
        np.testing.assert_array_equal(y,ytest)

if __name__== '__main__':
    unittest.main()

