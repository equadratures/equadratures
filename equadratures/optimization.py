import numpy as np
from scipy import optimize
from .poly import Poly

class Optimization:
    
    def __init__(self,method='SLSQP'):
        self.method = method
        self.obj = None
        self.grad = None
        self.constraints = []
    
    def AddObjective(self,objDict):
        assert 'poly' in objDict
        Poly = objDict['poly']
        obj = Poly.getPolyFitFunction()
        grad = Poly.getPolyGradFitFunction()
        if 'U' in objDict:
            U = objDict['U']
            self.obj = lambda x: np.array(obj(np.dot(x.reshape(1,-1),U))).flatten()
            self.grad = lambda x: np.dot(U,np.array(grad(np.dot(x.reshape(1,-1),U))).flatten())
        else:
            self.obj = lambda x: np.array(obj(x.reshape(1,-1))).flatten()
            self.grad = lambda x: np.array(grad(x.reshape(1,-1))).flatten()
        
        #TODO: Add a getPolyHessFitFunction()
        
        return None
    
    def AddLinearIneqCon(self,A,b_l,b_u):
        self.constraints.append({'type':'ineq', 'fun': lambda x: np.dot(A,x) - b_l, 'jac': lambda x: A})
        self.constraints.append({'type':'ineq', 'fun': lambda x: -np.dot(A,x) + b_u, 'jac': lambda x: -A})
        return None
    
    def AddNonLinearIneqCon(self,conDict): 
        assert 'poly','bounds' in conDict
        Poly = conDict['poly']
        bounds = conDict['bounds']
        g = Poly.getPolyFitFunction()
        grad = Poly.getPolyGradFitFunction()
        if 'U' in conDict:
            U = conDict['U']
            self.constraints.append({'type':'ineq', 'fun': lambda x: np.array(g(np.dot(x.reshape(1,-1),U))-bounds[0]).flatten(), \
                                     'jac': lambda x: np.dot(U,np.array(grad(np.dot(x.reshape(1,-1),U))).flatten())})
            self.constraints.append({'type':'ineq', 'fun': lambda x: np.array(-g(np.dot(x.reshape(1,-1),U))+bounds[1]).flatten(), \
                                     'jac': lambda x: -np.dot(U,np.array(grad(np.dot(x.reshape(1,-1),U))).flatten())})
        else:
            self.constraints.append({'type':'ineq', 'fun': lambda x: np.array(g(x.reshape(1,-1))-bounds[0]).flatten(), \
                                     'jac': lambda x: np.array(grad(x.reshape(1,-1))).flatten()})
            self.constraints.append({'type':'ineq', 'fun': lambda x: np.array(-g(x.reshape(1,-1))+bounds[1]).flatten(), \
                                     'jac': lambda x: np.array(-grad(x.reshape(1,-1))).flatten()})
        return None
    
    def AddLinearEqCon(self,A,b):
        assert self.method is not 'COBYLA'
        self.constraints.append({'type':'eq', 'fun': lambda x: np.dot(A,x) - b, 'jac': lambda x: A})
        return None
    
    def AddNonLinearEqCon(self,conDict):
        assert 'poly','val' in conDict
        assert self.method is not 'COBYLA'
        Poly = conDict['poly']
        val = conDict['val']
        g = Poly.getPolyFitFunction()
        grad = Poly.getPolyGradFitFunction()
        if 'U' in conDict:
            U = conDict['U']
            self.constraints.append({'type':'eq', 'fun': lambda x: np.array(g(np.dot(x.reshape(1,-1),U))-val).flatten(), \
                                     'jac': lambda x: np.dot(U,np.array(grad(np.dot(x.reshape(1,-1),U))).flatten())})
        else:
            self.constraints.append({'type':'eq', 'fun': lambda x: np.array(g(x.reshape(1,-1))-val).flatten(), \
                                     'jac': lambda x: np.array(grad(x.reshape(1,-1))).flatten()})
        return None
    
    def OptimizePoly(self,x0):
        if self.constraints:
            sol = optimize.minimize(self.obj, x0, method=self.method,jac=self.grad,constraints=self.constraints,options={'disp': False})
        return sol