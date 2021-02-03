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
        X,y = datasets.gen_linear(n_observations=500,bias=5,n_dim=10,random_seed=42)
        ynoise = y + np.random.RandomState(42).normal(0,0.1,500).reshape(-1,1)

        score = datasets.score(y,ynoise,'r2')
        np.testing.assert_array_almost_equal(score, 0.928, decimal=3, err_msg='Problem!')
        score = datasets.score(y,ynoise,'adjusted_r2',X=X)
        np.testing.assert_array_almost_equal(score, 0.927, decimal=3, err_msg='Problem!')
        score = datasets.score(y,ynoise,'mae')
        np.testing.assert_array_almost_equal(score, 0.078, decimal=3, err_msg='Problem!')
        score = datasets.score(y,ynoise,'normalised_mae')
        np.testing.assert_array_almost_equal(score, 0.220, decimal=3, err_msg='Problem!')
        score = datasets.score(y,ynoise,'rmse')
        np.testing.assert_array_almost_equal(score, 0.098, decimal=3, err_msg='Problem!')

    def test_dataloader(self):
        """
        Test the load_eq_dataset() function.

        NOTE: This adds an external dependency to the EQ tests, since it relies on the data-sets repo.
        """
        dataset = 'naca0012'
        data = datasets.load_eq_dataset(dataset)
        self.assertIsInstance(data,np.lib.npyio.NpzFile)

if __name__== '__main__':
    unittest.main()

