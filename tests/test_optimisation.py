import numpy as np
import scipy as sp
from unittest import TestCase
import unittest
import equadratures as eq

import nlopt

import warnings
warnings.filterwarnings('ignore')

class Test_optimisation(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.degf = 4
        cls.degg1 = 1
        cls.degg2 = 3
        cls.valg2 = -1.0

    @staticmethod
    def ObjFun1(x):
        if x.ndim == 1:
            return sp.optimize.rosen(x)
        else:
            f = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                f[i] = sp.optimize.rosen(x[i,:])
            return f
    
    @staticmethod
    def ObjFun2(s):
        n = s.size
        f = 0
        for i in range(n):
            f += 0.5 * (s[i]**4 - 16.0*s[i]**2 + 5.0*s[i])
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
        return g2_Hess

    @staticmethod
    def ConFun2(x):
        g2 = np.zeros((x.shape[0]))
        for i in range(g2.shape[0]):
            g2[i] = x[i,0]**3 - x[i,1]
        return g2

    def test_optimise_unconstrained_poly(self):
        n = 2
        N = 20
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        for method in ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B', 'Powell', 'Nelder-Mead', 'trust-ncg']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(fpoly)
            x0 = np.random.uniform(-1.0, 1.0, n)
            sol = Opt.optimise(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    def test_optimise_custom_function_bounds_maximise(self):
        n = 2

        for method in ['L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(custom={'function': lambda x: self.ConFun1(x.reshape(1,-1))}, maximise=True)
            Opt.add_bounds(-np.ones(n), np.ones(n))
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    def test_optimise_custom_function_linear_ineq_con1(self):
        n = 2
        N = 20
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        for method in ['COBYLA', 'SLSQP', 'trust-constr']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_linear_ineq_con(np.eye(n), -np.inf*np.ones(n), np.ones(n))
            x0 = np.random.uniform(-1.0, 1.0, n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    def test_optimise_custom_function_linear_ineq_con2(self):
        n = 2
        N = 20
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        for method in ['COBYLA', 'SLSQP', 'trust-constr']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_linear_ineq_con(np.eye(n), -np.ones(n), np.inf*np.ones(n))
            x0 = np.random.uniform(-1.0, 1.0, n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    def test_optimise_constrained_poly1(self):
        n = 2
        N = 20
        bounds = [-np.inf,2.0]
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis,  method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        # Active subspace and values for g1
        g1 = self.ConFun1(X)
        g1param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degg1)
        g1Parameters = [g1param for i in range(n)]
        myBasis = eq.Basis('total-order')
        g1poly = eq.Poly(g1Parameters, myBasis,  method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':g1})
        g1poly.set_model()
        for method in ['trust-constr', 'SLSQP', 'COBYLA']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_bounds(-np.ones(n), np.ones(n))
            Opt.add_nonlinear_ineq_con({'poly': g1poly, 'bounds': bounds})
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_constrained_poly2(self):
        n = 2
        N = 20
        bounds = [0.0,np.inf]
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis,  method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        # Active subspace and values for g1
        g1 = self.ConFun1(X)
        g1param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degg1)
        g1Parameters = [g1param for i in range(n)]
        myBasis = eq.Basis('total-order')
        g1poly = eq.Poly(g1Parameters, myBasis,  method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':g1})
        g1poly.set_model()
        for method in ['trust-constr', 'SLSQP', 'COBYLA']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_bounds(-np.ones(n), np.ones(n))
            Opt.add_nonlinear_ineq_con({'poly': g1poly, 'bounds': bounds})
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_ineq_constrained_function1(self):
        n = 2
        N = 20
        bounds = [-np.inf,2.0]
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        g1Func = lambda x: bounds[1] - self.ConFun1(x.reshape(1,-1))
        g1Grad = lambda x: -self.ConFun1_Deriv(x.flatten())
        g1Hess = lambda x: -self.ConFun1_Hess(x.flatten())

        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_nonlinear_ineq_con(custom={'function': g1Func, 'jac_function': g1Grad, 'hess_function': g1Hess})
            Opt.add_linear_eq_con(np.eye(n), np.ones(n))
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_ineq_constrained_function2(self):
        n = 2
        N = 20
        bounds = [-np.inf,2.0]
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        g1Func = lambda x: bounds[1] - self.ConFun1(x.reshape(1,-1))

        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_nonlinear_ineq_con(custom={'function': g1Func})
            Opt.add_linear_eq_con(np.eye(n), np.ones(n))
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_eq_constrained_function1(self):
        n = 2
        N = 20
        value = 2.0
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        g1Func = lambda x: value - self.ConFun1(x.reshape(1,-1))
        g1Grad = lambda x: -self.ConFun1_Deriv(x.flatten())
        g1Hess = lambda x: -self.ConFun1_Hess(x.flatten())

        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_nonlinear_eq_con(custom={'function': g1Func, 'jac_function': g1Grad, 'hess_function': g1Hess})
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    def test_optimise_eq_constrained_function2(self):
        n = 2
        N = 20
        value = 2.0
        X = np.random.uniform(-1.0, 1.0, (N, n))
        # Function values for f and Poly object
        f = self.ObjFun1(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('total-order')
        fpoly = eq.Poly(fParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':f})
        fpoly.set_model()
        g1Func = lambda x: value - self.ConFun1(x.reshape(1,-1))

        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimisation(method=method)
            Opt.add_objective(poly=fpoly)
            Opt.add_nonlinear_eq_con(custom={'function': g1Func})
            x0 = np.zeros(n)
            sol = Opt.optimise(x0)
            if sol['status'] == 0:
                np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)

    def test_optimise_trustregion(self):
        n = 2
        Opt = eq.Optimisation(method='trust-region')
        Opt.add_objective(custom={'function': self.ObjFun1})
        x0 = np.zeros(n)
        sol = Opt.optimise(x0, del_k=0.5)
        np.testing.assert_almost_equal(sol['fun'], 0.0, decimal=6)

    def test_optimise_trustregion_random(self):
        n = 2
        Opt = eq.Optimisation(method='trust-region')
        Opt.add_objective(custom={'function': self.ObjFun1})
        x0 = np.zeros(n)
        sol = Opt.optimise(x0, del_k=0.5, random_initial=True)
        np.testing.assert_almost_equal(sol['fun'], 0.0, decimal=6)
         
    def test_optimise_trustregion_bounds(self):
        n = 2
        Opt = eq.Optimisation(method='trust-region')
        Opt.add_objective(custom={'function': self.ObjFun1})
        Opt.add_bounds(-np.ones(n), np.ones(n))
        x0 = np.zeros(n)
        sol = Opt.optimise(x0, del_k=0.5)
        np.testing.assert_almost_equal(sol['fun'], 0.0, decimal=6)
            
    def test_optimise_omorf_vp(self):
        n = 10
        Opt = eq.Optimisation(method='omorf')
        Opt.add_objective(custom={'function': self.ObjFun2})
        x0 = -2*np.ones(n)
        sol = Opt.optimise(x0, del_k=0.5)
        np.testing.assert_almost_equal(sol['fun'], -39.166165*n, decimal=4)
            
    def test_optimise_omorf_as(self):
        n = 10
        Opt = eq.Optimisation(method='omorf')
        Opt.add_objective(custom={'function': self.ObjFun2})
        x0 = -2*np.ones(n)
        sol = Opt.optimise(x0, del_k=0.5, subspace_method='active-subspaces')
        np.testing.assert_almost_equal(sol['fun'], -39.166165*n, decimal=4)
            
    def test_optimise_omorf_bounds(self):
        n = 10
        Opt = eq.Optimisation(method='omorf')
        Opt.add_objective(custom={'function': self.ObjFun2})
        Opt.add_bounds(-5.12*np.ones(n), 5.12*np.ones(n))
        x0 = -2*np.ones(n)
        sol = Opt.optimise(x0, del_k=0.5)
        np.testing.assert_almost_equal(sol['fun'], -39.166165*n, decimal=4)

if __name__ == '__main__':
    unittest.main()
