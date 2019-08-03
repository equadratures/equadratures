import numpy as np
import scipy as sp
import unittest
import equadratures as eq

class test_optimization(unittest.TestCase):
    
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
    
    @staticmethod
    def ObjFun_Subspace(x,a):
        u = np.dot(x,a)
        f = np.zeros((u.shape[0]))
        for i in range(u.shape[0]):
            f[i] = sp.optimize.rosen(u[i,:])
        return u, f 
    
    @staticmethod    
    def ConFun1_Subspace(x,a):
        w = np.dot(x,a)
        g1 = np.zeros((w.shape[0]))
        for i in range(g1.shape[0]):
            g1[i] = w[i,0] + w[i,1]
        return w, g1
    
    @staticmethod
    def ConFun2_Subspace(x,a):
        v = np.dot(x,a)
        g2 = np.zeros((v.shape[0]))
        for i in range(g2.shape[0]):
            g2[i] = v[i,0]**3 - v[i,1]
        return v, g2 
    
    def test_optimizePoly_unconstrained_poly(self):
        n = 2
        N = 20
        
        X = np.random.uniform(-1.0, 1.0, (N, n))
#       Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('Total order')
        fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=X, training_outputs=f)
        
        for method in ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B', 'Powell', 'Nelder-Mead', 'trust-ncg']:
            Opt = eq.Optimization(method=method)
            Opt.addObjective(Poly=fpoly)
            x0 = np.random.uniform(-1.0, 1.0, n)
            sol = Opt.optimizePoly(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=3)
    
    def test_optimizePoly_unconstrained_poly_subspace(self):
        df = 2
        
        n = 50
        N = 5000
        
        X = np.random.uniform(-1.0, 1.0, (N, n))
#       Active subspace and values for f
        U = sp.linalg.orth(np.random.rand(n,df))
        u, f = self.ObjFun_Subspace(X,U)
        fparam = eq.Parameter(distribution='uniform', lower=-6., upper=6., order=self.degf)
        fParameters = [fparam for i in range(df)]
        myBasis = eq.Basis('Total order')
        fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=u, training_outputs=f)
        
        for method in ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B', 'Powell', 'Nelder-Mead', 'trust-ncg']:
            Opt = eq.Optimization(method=method)
            Opt.addObjective(Poly=fpoly, subspace=U)
            x0 = np.zeros(n)
            sol = Opt.optimizePoly(x0)
            np.testing.assert_almost_equal(np.dot(sol['x'], U).flatten(), np.array([1.0, 1.0]), decimal=3)
    
    def test_optimizePoly_constrained_poly(self):
        n = 2
        N = 20
        
        X = np.random.uniform(-1.0, 1.0, (N, n))
#       Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('Total order')
        fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=X, training_outputs=f)
#       Active subspace and values for g1
        g1 = self.ConFun1(X)
        g1param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degg1)
        g1Parameters = [g1param for i in range(n)]
        myBasis = eq.Basis('Total order')
        g1poly = eq.Polyreg(g1Parameters, myBasis, training_inputs=X, training_outputs=g1)
        
        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimization(method=method)
            Opt.addObjective(Poly=fpoly)
            Opt.addLinearIneqCon(np.eye(n), -np.ones(n), np.ones(n))
            Opt.addNonLinearIneqCon(self.boundsg1, Poly=g1poly)
            x0 = np.zeros(n)
            sol = Opt.optimizePoly(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    
    def test_optimizePoly_constrained_function(self):
        n = 2
        N = 20
        
        X = np.random.uniform(-1.0, 1.0, (N, n))
#       Function values for f and Poly object
        f = self.ObjFun(X)
        fparam = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=self.degf)
        fParameters = [fparam for i in range(n)]
        myBasis = eq.Basis('Total order')
        fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=X, training_outputs=f)
        
        g1Func = lambda x: self.ConFun1(x.reshape(1,-1))
        g1Grad = lambda x: self.ConFun1_Deriv(x.flatten())
        g1Hess = lambda x, v: self.ConFun1_Hess(x.flatten())
        
        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimization(method=method)
            Opt.addObjective(Poly=fpoly)
            Opt.addNonLinearIneqCon(self.boundsg1, Function=g1Func, jacFunction=g1Grad, hessFunction=g1Hess)
            Opt.addLinearEqCon(np.eye(n), np.ones(n))
            x0 = np.zeros(n)
            sol = Opt.optimizePoly(x0)
            np.testing.assert_almost_equal(sol['x'].flatten(), np.array([1.0, 1.0]), decimal=2)
    
    def test_optimizePoly_constrained_poly_subspace(self):
        df = 2
        dg1 = 2 
        dg2 = 2
        
        n = 50
        N = 5000
        
        X = np.random.uniform(-1.0, 1.0, (N, n))
#       Active subspace and values for f
        U = sp.linalg.orth(np.random.rand(n,df))
        u, f = self.ObjFun_Subspace(X,U)
        fparam = eq.Parameter(distribution='uniform', lower=-6., upper=6., order=self.degf)
        fParameters = [fparam for i in range(df)]
        myBasis = eq.Basis('Total order')
        fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=u, training_outputs=f)
        
#       Active subspace and values for g1
        W = sp.linalg.orth(np.random.rand(n,dg1))
        w, g1 = self.ConFun1_Subspace(X,W)
        g1param = eq.Parameter(distribution='uniform', lower=-6., upper=6., order=self.degg1)
        g1Parameters = [g1param for i in range(dg1)]
        myBasis = eq.Basis('Total order')
        g1poly = eq.Polyreg(g1Parameters, myBasis, training_inputs=w, training_outputs=g1)
        
#       Active subspace and values for g2
        V = sp.linalg.orth(np.random.rand(n,dg2))
        v, g2 = self.ConFun2_Subspace(X,V)
        g2param = eq.Parameter(distribution='uniform', lower=-6., upper=6., order=self.degg2)
        g2Parameters = [g2param for i in range(dg2)]
        myBasis = eq.Basis('Total order')
        g2poly = eq.Polyreg(g2Parameters, myBasis, training_inputs=v, training_outputs=g2)   
        
        for method in ['trust-constr', 'SLSQP']:
            Opt = eq.Optimization(method=method)
            Opt.addObjective(Poly=fpoly, subspace=U)
            Opt.addBounds(-np.ones(n), np.ones(n))
            Opt.addNonLinearIneqCon(self.boundsg1, Poly=g1poly, subspace=W)
            Opt.addNonLinearEqCon(self.valg2, Poly=g2poly, subspace=V)
            x0 = np.zeros(n)
            sol = Opt.optimizePoly(x0)
            np.testing.assert_almost_equal(np.dot(sol['x'], U).flatten(), np.array([1.0, 1.0]), decimal=3)

if __name__ == '__main__':  
    unittest.main()
    
