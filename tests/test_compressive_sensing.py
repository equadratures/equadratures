from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import scipy.stats as st
class TestB(TestCase):

    def test_cs(self):
        repo = np.DataSource('.')
        file_object = repo.open('https://raw.githubusercontent.com/psesh/turbodata/master/three_blades/blade_A/design_parameters.dat')
        X = np.loadtxt(file_object)
        Y = np.loadtxt("h_Y.dat")
        N = X.shape[0]
        p_order = 2
        params = []
        basis_orders = []

        for i in range(25):
                params.append(Parameter(p_order, distribution = 'Custom', data = np.reshape(X[:,i], (N,))))
                basis_orders.append(p_order)

        basis = Basis("total-order", orders = basis_orders)
        num_obs = 200
        chosen_points = np.random.choice(range(N), size = num_obs, replace = False)
        X_red = X[chosen_points,:]
        Y_red = Y[chosen_points]
        remaining_pts = np.delete(np.arange(N), chosen_points)
        chosen_valid_pts = np.random.choice(remaining_pts, size = 30, replace = False)
        x_eval = X[chosen_valid_pts]

        poly = Poly(params, basis, method='compressive-sensing', sampling_args={'sample-points':X_red, 'sample-outputs':Y_red})
        poly.set_model()
        y_eval = poly.get_polyfit(X)
        y_valid = Y
        a,b,r,_,_ = st.linregress(y_eval.flatten(),y_valid.flatten())
        r2 = np.round(r**2, 4)
        np.testing.assert_array_less(0.88, r2, err_msg='Problem!')

if __name__== '__main__':
    unittest.main()