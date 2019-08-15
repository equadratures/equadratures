import numpy as np
import scipy as sp
from unittest import TestCase
import unittest
import equadratures as eq

class Test_optimisation(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.degf = 4
        cls.degg1 = 1
        cls.degg2 = 3
        cls.boundsg1 = [-np.inf,2.0]
        cls.valg2 = -1.0

    @staticmethod
    def ObjFun(x):
        f = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            f[i] = sp.optimize.rosen(x[i,:])
        return f

    @staticmethod
    def ConFun1(x):
        g1 = np.zeros((x.shape[0]))
        for i in range(g1.shape[0]):
            g1[i] = x[i,0] + x[i,1]
        return g1

    @staticmethod
    def ConFun1_Deriv(x):
        g2_Deriv = np.zeros(2)
        g2_Deriv[0] = 1.0
        g2_Deriv[1] = 1.0
        return g2_Deriv

    @staticmethod
    def ConFun1_Hess(x):
        g2_Hess = np.zeros((2, 2))
        g2_Hess[0, 0] = 0.0
        g2_Hess[0, 1] = 0.0
        g2_Hess[1, 0] = 0.0
        g2_Hess[1, 1] = 0.0
        return g2_Hess

    @staticmethod
    def ConFun2(x):
        g2 = np.zeros((x.shape[0]))
        for i in range(g2.shape[0]):
            g2[i] = x[i,0]**3 - x[i,1]
        return g2

    def test_optimise_poly_unconstrained_poly(self):
        n = 2
        N = 20

        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        for method in ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B', 'Powell', 'Nelder-Mead', 'trust-ncg']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(fpoly)
            x0 = np.random.uniform(-1.0, 1.0, n)
            sol = Opt.optimise_poly(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    def test_optimise_poly_constrained_poly(self):
        n = 2
        N = 20
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis,  method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        # Active subspace and values for g1
        g1 = self.ConFun1(X)
        g1param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degg1)
        g1Parameters = [g1param for i in range(n)]
        myBasis = eq.Basis('total-order')
        g1poly = eq.Poly(g1Parameters, myBasis,  method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':g1})
        g1poly.set_model()
        for method in ['trust-constr', 'SLSQP', 'COBYLA']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_bounds(-np.ones(n), np.ones(n))
            Opt.add_nonlinear_ineq_con({'poly': g1poly, 'bounds': self.boundsg1})
            x0 = np.zeros(n)
            sol = Opt.optimise_poly(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_poly_constrained_function(self):
        n = 2
        N = 20
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        g1Func = lambda x: self.boundsg1[1] - self.ConFun1(x.reshape(1,-1))
        g1Grad = lambda x: -self.ConFun1_Deriv(x.flatten())
        g1Hess = lambda x: -self.ConFun1_Hess(x.flatten())

        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_nonlinear_ineq_con(custom={'function': g1Func, 'jac_function': g1Grad, 'hess_function': g1Hess})
            Opt.add_linear_eq_con(np.eye(n), np.ones(n))
            x0 = np.zeros(n)
            sol = Opt.optimise_poly(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)

if __name__ == '__main__':
    unittest.main()
