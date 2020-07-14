from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from scipy.stats import skew, linregress, multivariate_normal

def fun(x):
    return 5.0 * x[0] ** 3 - x[0] * x[1] + 3.0 * x[1] * x[2] ** 3 + 32.0

class TestF(TestCase):
    def test_nataf(self):
        np.random.seed(1)
        zeta_1 = Parameter(distribution='truncated-gaussian', shape_parameter_A=3.0, shape_parameter_B=2.0,
                           order=15, lower=-2.0, upper=4.0)
        zeta_2 = Parameter(distribution='truncated-gaussian', shape_parameter_A=-1.0, shape_parameter_B=0.1,
                           order=15, lower=-5.0, upper=5.0)
        zeta_3 = Parameter(distribution='truncated-gaussian', shape_parameter_A=2.0, shape_parameter_B=2.0,
                           order=15, lower=0.0, upper=6.0)
        R = np.eye(3)
        R[0, 1] = 0.6
        R[0, 2] = 0.3
        R[2, 1] = 0.2
        R[1, 0] = R[0, 1]
        R[2, 0] = R[0, 2]
        R[1, 2] = R[2, 1]
        myBasis = Basis('tensor-grid')
        myPoly = Poly([zeta_1, zeta_2, zeta_3], myBasis, method='numerical-integration')
        myNataf = Correlations(R, poly=myPoly, method='nataf-transform')
        samples_mc = myNataf.get_correlated_samples(N=50000)
        f_mc = evaluate_model(samples_mc, fun)
        myNataf.set_model(fun)
        myTransformedPoly = myNataf.get_transformed_poly()
        mean, variance = myTransformedPoly.get_mean_and_variance()
        skewness, kurtosis = myTransformedPoly.get_skewness_and_kurtosis()

        np.testing.assert_almost_equal(mean / np.mean(f_mc), 1.0, decimal=1.5)
        np.testing.assert_almost_equal(variance / np.var(f_mc), 1.0, decimal=1.5)

        np.testing.assert_almost_equal(skewness / skew(f_mc)[0], 1.0, decimal=1.5)

    def test_GS(self):
        np.random.seed(1)
        zeta_1 = Parameter(distribution='truncated-gaussian', shape_parameter_A=3.0, shape_parameter_B=2.0,
                           order=5, lower=-2.0, upper=4.0)
        zeta_2 = Parameter(distribution='truncated-gaussian', shape_parameter_A=-1.0, shape_parameter_B=0.1,
                           order=5, lower=-5.0, upper=5.0)
        zeta_3 = Parameter(distribution='truncated-gaussian', shape_parameter_A=2.0, shape_parameter_B=2.0,
                           order=5, lower=0.0, upper=6.0)
        R = np.eye(3)
        R[0, 1] = 0.6
        R[0, 2] = 0.3
        R[2, 1] = 0.2
        R[1, 0] = R[0, 1]
        R[2, 0] = R[0, 2]
        R[1, 2] = R[2, 1]
        myBasis = Basis('tensor-grid')
        myPoly = Poly([zeta_1, zeta_2, zeta_3], myBasis, method='least-squares',
                      sampling_args={'mesh': 'monte-carlo', 'subsampling-algorithm': 'lu'})
        myGS = Correlations(R, poly=myPoly, method='gram-schmidt')
        samples_mc = myGS.get_correlated_samples(N=500)
        f_mc = evaluate_model(samples_mc, fun)
        myGS.set_model(fun)
        myTransformedPoly = myGS.get_transformed_poly()

        s, _, r, _, _ = linregress(myTransformedPoly.get_polyfit(samples_mc).reshape(-1),
                                   f_mc.reshape(-1))
        np.testing.assert_almost_equal(s, 1.0, decimal=2)
        np.testing.assert_almost_equal(r, 1.0, decimal=2)

    def test_pdf(self):
        np.random.seed(1)
        X_test = np.random.uniform(-3, 3, (1000, 2))
        my_params = [Parameter(distribution='gaussian', shape_parameter_A=0, shape_parameter_B=1.0,
                               order=3) for _ in range(2)]
        corr_mat = np.eye(2)
        corr_mat[0,1] = corr_mat[1,0] = 0.5
        my_corr = Correlations(corr_mat, parameters=my_params)
        test_pdf = my_corr.get_pdf(X_test)
        truth_pdf = multivariate_normal.pdf(X_test, mean=[0,0], cov=corr_mat)
        np.testing.assert_almost_equal(np.linalg.norm(test_pdf - truth_pdf), 0.0, decimal=2)

if __name__ == '__main__':
    unittest.main()
