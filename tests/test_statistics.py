from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from scipy.stats import skew, kurtosis

def rosenbrock_fun(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def phi(x):
    return np.sqrt(3) * x
def fun(X):
    x = phi(X)
    return 0.1 + 0.2 * x[0] + 0.3 * x[1] * x[2] + 0.4 * x[1] * x[2] * x[3] + 0.5 * x[0] * x[1] * x[2] * x[3]
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
    def test_higher_order_statistics(self):
        np.random.seed(0)
        p_order = 4
        my_basis = Basis('total-order', orders=[p_order, p_order, p_order, p_order])
        my_params = [Parameter(order=p_order, distribution='uniform', lower=-1, upper=1) for _ in range(4)]
        X = np.random.uniform(-1, 1, (10000, 4))
        y = evaluate_model(X, fun)
        my_poly = Poly(my_params, my_basis,  method='least-squares', sampling_args={'sample-inputs':X, 'sample-outputs': y})
        my_poly.set_model(fun)
        mean, variance = my_poly.get_mean_and_variance()
        skewness, kurtosis = my_poly.get_skewness_and_kurtosis()
        np.testing.assert_almost_equal(mean, 0.1, decimal=5)
        analytical_variance = 0.2**2 + 0.3**2 + 0.4**2 + 0.5**2 # == 0.54
        np.testing.assert_almost_equal(variance, 0.54, decimal=5)
        np.testing.assert_almost_equal(skewness, 0.60481, decimal=5)
        np.testing.assert_almost_equal(kurtosis, 10.69801, decimal=5)
        condskew4 = my_poly.get_conditional_skewness_indices(4)
        np.testing.assert_almost_equal(condskew4[(0, 1, 2, 3)],  1.0, decimal=5)
        condkurt1 = my_poly.get_conditional_kurtosis_indices(1)
        condkurt2 = my_poly.get_conditional_kurtosis_indices(2)
        condkurt3 = my_poly.get_conditional_kurtosis_indices(3)
        # Verification required!
        np.testing.assert_almost_equal(condkurt1[(0, )],  0.0009232, decimal=5)
        np.testing.assert_almost_equal(condkurt2[(1, 2)], 0.008413, decimal=5)
        np.testing.assert_almost_equal(condkurt3[(0,1,2)], 0.006924, decimal=5)
        np.testing.assert_almost_equal(condkurt3[(1,2,3)], 0.137596, decimal=5)
    def test_total_sobol_indices(self):
        order_parameters = 3
        mass = Parameter(distribution='uniform', lower=30.0, upper=60.0, order=order_parameters)
        area = Parameter(distribution='uniform', lower=0.005, upper=0.020, order=order_parameters)
        volume = Parameter(distribution='uniform', lower=0.002, upper=0.010, order=order_parameters)
        spring = Parameter(distribution='uniform', lower=1000., upper=5000., order=order_parameters)
        pressure = Parameter(distribution='uniform', lower=90000., upper=110000., order=order_parameters)
        ambtemp = Parameter(distribution='uniform', lower=290., upper=296., order=order_parameters)
        gastemp = Parameter(distribution='uniform', lower=340., upper=360., order=order_parameters)
        parameters = [mass, area, volume, spring, pressure, ambtemp, gastemp]

        def piston(x):
            mass, area, volume, spring, pressure, ambtemp, gastemp = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
            A = pressure * area + 19.62*mass - (spring * volume)/(1.0 * area)
            V = (area/(2*spring)) * ( np.sqrt(A**2 + 4*spring * pressure * volume * ambtemp/gastemp) - A)
            C = 2 * np.pi * np.sqrt(mass/(spring + area**2 * pressure * volume * ambtemp/(gastemp * V**2)))
            return C
        mybasis = Basis('total-order')
        pistonmodel = Poly(parameters, mybasis, method='least-squares', \
                    sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'qr', \
                            'sampling-ratio':1.0})
        pts_for_evaluation = pistonmodel.get_points()
        model_evals = evaluate_model(pts_for_evaluation, piston)
        pistonmodel.set_model(model=model_evals)
        all_indices = pistonmodel.get_total_sobol_indices()
        np.testing.assert_array_less(all_indices[0], all_indices[1])
if __name__== '__main__':
    unittest.main()