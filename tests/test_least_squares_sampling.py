from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
def fun(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
class TestC(TestCase):
    def test_qr(self):
        zeta_1 = Parameter(distribution='uniform', order=4, lower= -2.0, upper=2.0)
        zeta_2 = Parameter(distribution='uniform', order=4, lower=-1.0, upper=3.0)
        myBasis1 = Basis('tensor-grid')
        myBasis2 = Basis('total-order')
        myPoly1 = Poly([zeta_1, zeta_2], myBasis1, method='numerical-integration')
        myPoly2 = Poly([zeta_1, zeta_2], myBasis2, method='least-squares', sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'qr', 'sampling-ratio':1.0})
        myPoly1.set_model(fun)
        myPoly2.set_model(fun)
        pts1 = myPoly1.get_points()
        pts2 = myPoly2.get_points()
        N = 30
        z1 = np.linspace(zeta_1.lower, zeta_1.upper, N)
        z2 = np.linspace(zeta_2.lower, zeta_2.upper, N)
        [Z1, Z2] = np.meshgrid(z1, z2)
        Z1_vec = np.reshape(Z1, (N*N, 1))
        Z2_vec = np.reshape(Z2, (N*N, 1))
        samples = np.hstack([Z1_vec, Z2_vec])
        PolyApprox1 = myPoly1.get_polyfit( samples )
        PolyApprox1 = np.reshape(PolyApprox1, (N, N))
        PolyApprox2 = myPoly2.get_polyfit( samples )
        PolyApprox2 = np.reshape(PolyApprox2, (N, N))
        np.testing.assert_array_almost_equal(PolyApprox1, PolyApprox2, decimal=7, err_msg='Problem!')
    def test_newton_svd(self):
        zeta_1 = Parameter(distribution='uniform', order=4, lower= -2.0, upper=2.0)
        zeta_2 = Parameter(distribution='uniform', order=4, lower=-1.0, upper=3.0)
        myBasis3 = Basis('total-order')
        myPoly3 = Poly([zeta_1, zeta_2], myBasis3, method='least-squares', sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'newton', 'sampling-ratio':1.0})
        pts3 = myPoly3.get_points()
        model_evals3 = evaluate_model(pts3, fun)
        myPoly3.set_model(model_evals3)
        myBasis4 = Basis('total-order')
        myPoly4 = Poly([zeta_1, zeta_2], myBasis3, method='least-squares', sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'svd', 'sampling-ratio':1.0})
        pts4 = myPoly4.get_points()
        model_evals4 = evaluate_model(pts4, fun)
        myPoly4.set_model(model_evals4)
        np.testing.assert_array_almost_equal(myPoly3.get_coefficients(), myPoly4.get_coefficients(), decimal=8, err_msg='Problem!')
    def test_least_squares_verbose(self):
        zeta_1 = Parameter(distribution='uniform', order=4, lower= -2.0, upper=2.0)
        zeta_2 = Parameter(distribution='uniform', order=4, lower=-1.0, upper=3.0)
        myBasis3 = Basis('total-order')
        myPoly3 = Poly([zeta_1, zeta_2], myBasis3, method='least-squares', \
            sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'newton', 'sampling-ratio':1.0}, \
            solver_args={'verbose': True})
        pts3 = myPoly3.get_points()
        model_evals3 = evaluate_model(pts3, fun)
        myPoly3.set_model(model_evals3)
if __name__== '__main__':
    unittest.main()