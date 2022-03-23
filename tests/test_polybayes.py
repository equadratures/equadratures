from unittest import TestCase
import unittest
import equadratures as eq
import numpy as np
import scipy.stats as st


def f(x):
    return np.exp(np.sum(x))


class TestPolybayes(TestCase):

    def setUp(self) -> None:
        s1 = eq.Parameter(distribution='uniform', lower=-1, upper=1, order=5,endpoints='both')
        s2 = eq.Parameter(distribution='uniform', lower=-1, upper=1, order=5,endpoints='both')
        my_params = [s1, s2]
        my_basis = eq.Basis('total-order', orders=[5, 5])
        self.cardinality = my_basis.get_cardinality()
        self.my_polybayes = eq.Polybayes(my_params, my_basis, sigma_data=0.01)
        self.my_rng = np.random.default_rng(0)
        self.N_train = 19
        self.N_test = 10

    def test_fitting(self):
        input_training_data = self.my_rng.uniform(-1, 1, size=(self.N_train, 2))
        output_training_data = eq.evaluate_model(input_training_data, f).reshape(-1)
        self.my_polybayes.compute_posterior_coefficients(input_training_data, output_training_data)

        input_test_data = self.my_rng.uniform(-1, 1, size=(self.N_test, 2))
        output_test_data = eq.evaluate_model(input_test_data, f).reshape(-1)
        mean_pred, _ = self.my_polybayes.get_posterior_fit(input_test_data)
        r2 = st.linregress(mean_pred, output_test_data)[2]**2

        np.testing.assert_array_less(0.80, r2, err_msg='Polybayes r2 too low.')

    def test_condition_on_mean(self):
        input_training_data = self.my_rng.uniform(-1, 1, size=(self.N_train, 2))
        output_training_data = eq.evaluate_model(input_training_data, f).reshape(-1)
        self.my_polybayes.compute_posterior_coefficients(input_training_data, output_training_data)

        input_test_data = self.my_rng.uniform(-1, 1, size=(self.N_test, 2))
        output_test_data = eq.evaluate_model(input_test_data, f).reshape(-1)
        mean_pred, _ = self.my_polybayes.get_posterior_fit(input_test_data, estimated_mean=1.37)
        r2 = st.linregress(mean_pred, output_test_data)[2]**2

        np.testing.assert_array_less(0.80, r2, err_msg='Polybayes r2 too low.')

    def test_output_moments(self):
        input_training_data = self.my_rng.uniform(-1, 1, size=(self.N_train, 2))
        output_training_data = eq.evaluate_model(input_training_data, f).reshape(-1)
        self.my_polybayes.compute_posterior_coefficients(input_training_data, output_training_data)

        est_output_mean, est_output_std = self.my_polybayes.get_posterior_output_moments()
        print(est_output_mean, est_output_std)
        np.testing.assert_almost_equal(est_output_mean[0], 1.37, decimal=np.log10(est_output_mean[1]))
        np.testing.assert_almost_equal(est_output_std[0], 1.17, decimal=np.log10(est_output_std[1]))


#%%
if __name__== '__main__':
    unittest.main()

