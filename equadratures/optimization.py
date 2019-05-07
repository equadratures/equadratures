from scipy import optimize

class Optimization:
    
    def __init__(self,method='trust-constr'):
        self.method = method
        self.obj = None
        self.jac = None
        self.hess = None
        self.constraints = []
    
    def addLinearIneqCon(self,A,b_l,b_u):
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b_l,b_u))
        else:
            self.constraints.append({'type':'ineq', 'fun': lambda x: A.dot(x) - b_l, 'jac': lambda x: A})
            self.constraints.append({'type':'ineq', 'fun': lambda x: -A.dot(x) + b_u, 'jac': lambda x: -A})
        return None
    
    def addNonLinearIneqCon(self,conDict): 
        assert 'bounds' in conDict
        bounds = conDict['bounds']
        assert 'poly' or 'function' in conDict
        if 'poly' in conDict:
            Poly = conDict['poly']
            g = Poly.getPolyFitFunction()
            if self.method == 'trust-constr':
                jac = Poly.getPolyGradFitFunction()
                hess = Poly.getPolyHessFitFunction()
                if 'subspace' in conDict:
                    U = conDict['subspace']
                    self.constraints.append(optimize.NonlinearConstraint(lambda x: g(x.reshape(1,-1).dot(U))[0], \
                                             bounds[0], bounds[1], jac = lambda x: U.dot(jac(x.reshape(1,-1).dot(U))[:,0]), \
                                             hess = lambda x, v: U.dot(hess(x.reshape(1,-1).dot(U))[:,:,0]).dot(U.T)))
                else:
                    self.constraints.append(optimize.NonlinearConstraint(lambda x: g(x.reshape(1,-1))[0], \
                                             bounds[0], bounds[1], jac = lambda x: jac(x.reshape(1,-1))[:,0], \
                                             hess = lambda x, v: hess(x.reshape(1,-1))[:,:,0]))
            elif self.method == 'SLSQP':
                jac = Poly.getPolyGradFitFunction()
                if 'subspace' in conDict:
                    U = conDict['subspace']
                    self.constraints.append({'type':'ineq', 'fun': lambda x: g(x.reshape(1,-1).dot(U))[0]-bounds[0], \
                                             'jac': lambda x: U.dot(jac(x.reshape(1,-1).dot(U))[:,0])})
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -g(x.reshape(1,-1).dot(U))[0]+bounds[1], \
                                             'jac': lambda x: -U.dot(jac(x.reshape(1,-1).dot(U))[:,0])})
                else:
                    self.constraints.append({'type':'ineq', 'fun': lambda x: g(x.reshape(1,-1))[0]-bounds[0], \
                                             'jac': lambda x: jac(x.reshape(1,-1))[:,0]})
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -g(x.reshape(1,-1))[0]+bounds[1], \
                                             'jac': lambda x: -jac(x.reshape(1,-1))[:,0]})
            elif self.method == 'COBYLA':
                if 'subspace' in conDict:
                    U = conDict['subspace']
                    self.constraints.append({'type':'ineq', 'fun': lambda x: g(x.reshape(1,-1).dot(U))-bounds[0]})
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -g(x.reshape(1,-1).dot(U))+bounds[1]})
                else:
                    self.constraints.append({'type':'ineq', 'fun': lambda x: g(x.reshape(1,-1))-bounds[0]})
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -g(x.reshape(1,-1))+bounds[1]})
        elif 'function' in conDict:
            con = conDict['function']
            if 'jacFunction' in conDict:
                jac = conDict['jacFunction']
            else:
                jac = '2-point'
            if 'hessFunction' in conDict:
                hess = conDict['hessFunction']
            else:
                hess = '2-point'
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(con,bounds[0],bounds[1],jac=jac,hess=hess))
            else:
                self.constraints.append({'type':'ineq', 'fun': lambda x: con(x)-bounds[0], 'jac': jac, 'hess': hess})
                self.constraints.append({'type':'ineq', 'fun': lambda x: -con(x)+bounds[1], 'jac': jac, 'hess': hess})
        return None
    
    def addLinearEqCon(self,A,b):
        assert self.method != 'COBYLA'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b,b))
        elif self.method == 'SLSQP':
            self.constraints.append({'type':'eq', 'fun': lambda x: A.dot(x) - b, 'jac': lambda x: A})
        return None
    
    def addNonLinearEqCon(self,conDict):
        assert 'val' in conDict
        assert 'poly' or 'function' in conDict
        assert self.method != 'COBYLA'
        if 'poly' in conDict:
            Poly = conDict['poly']
            val = conDict['val']
            g = Poly.getPolyFitFunction()
            if self.method == 'trust-constr':
                jac = Poly.getPolyGradFitFunction()
                hess = Poly.getPolyHessFitFunction()
                if 'subspace' in conDict:
                    U = conDict['subspace']
                    self.constraints.append(optimize.NonlinearConstraint(lambda x: g(x.reshape(1,-1).dot(U))[0], \
                                             0.0, 0.0, jac = lambda x: U.dot(jac(x.reshape(1,-1).dot(U))[:,0]), \
                                             hess = lambda x, v: U.dot(hess(x.reshape(1,-1).dot(U))[:,:,0]).dot(U.T)))
                else:
                    self.constraints.append(optimize.NonlinearConstraint(lambda x: g(x.reshape(1,-1))[0], \
                                             0.0, 0.0, jac = lambda x: jac(x.reshape(1,-1))[:,0], \
                                             hess = lambda x, v: hess(x.reshape(1,-1))[:,:,0]))
            elif self.method == 'SLSQP':
                jac = Poly.getPolyGradFitFunction()
                if 'subspace' in conDict:
                    U = conDict['subspace']
                    self.constraints.append({'type':'eq', 'fun': lambda x: g(x.reshape(1,-1).dot(U))[0]-val, \
                                             'jac': lambda x: U.dot(jac(x.reshape(1,-1).dot(U))[:,0])})
                else:
                    self.constraints.append({'type':'eq', 'fun': lambda x: g(x.reshape(1,-1))[0]-val, \
                                             'jac': lambda x: jac(x.reshape(1,-1))[0]})
        elif 'function' in conDict:
            con = conDict['function']
            if 'jacFunction' in conDict:
                jac = conDict['jacFunction']
            else:
                jac = '2-point'
            if 'hessFunction' in conDict:
                hess = conDict['hessFunction']
            else:
                hess = '2-point'
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(con,0.0,0.0,jac=jac,hess=hess))
            else:
                self.constraints.append({'type':'eq', 'fun': con, 'jac': jac, 'hess': hess})
        return None
    
    def optimizePoly(self,objDict,x0):
        assert 'poly' or 'func' in objDict
        if 'poly' in objDict:
            if 'min' in objDict:
                if objDict['min'] == True:
                    k = 1.0
                elif objDict['min'] == False:
                    k = -1.0
            else:
                k = 1.0
            Poly = objDict['poly']
            obj = Poly.getPolyFitFunction()
            if self.method == 'SLSQP':
                jac = Poly.getPolyGradFitFunction()
                if 'subspace' in objDict:
                    U = objDict['subspace']
                    self.obj = lambda x: k*obj(x.reshape(1,-1).dot(U))[0]
                    self.jac = lambda x: k*U.dot(jac(x.reshape(1,-1).dot(U))[:,0])
                else:
                    self.obj = lambda x: k*obj(x.reshape(1,-1))[0]
                    self.jac = lambda x: k*jac(x.reshape(1,-1))[:,0]
            elif self.method == 'COBYLA':
                if 'subspace' in objDict:
                    U = objDict['subspace']
                    self.obj = lambda x: k*obj(x.reshape(1,-1).dot(U))[0]
                else:
                    self.obj = lambda x: k*obj(x.reshape(1,-1))[0]
            else:
                jac = Poly.getPolyGradFitFunction()
                hess = Poly.getPolyHessFitFunction()
                if 'subspace' in objDict:
                    U = objDict['subspace']
                    self.obj = lambda x: k*obj(x.reshape(1,-1).dot(U))[0]
                    self.jac = lambda x: k*U.dot(jac(x.reshape(1,-1).dot(U))[:,0])
                    self.hess = lambda x: k*U.dot(hess(x.reshape(1,-1).dot(U))[:,:,0]).dot(U.T)
                else:
                    self.obj = lambda x: k*obj(x.reshape(1,-1))[0]
                    self.jac = lambda x: k*jac(x.reshape(1,-1))[:,0]
                    self.hess = lambda x: k*hess(x.reshape(1,-1))[:,:,0]
        elif 'function' in objDict:
            self.obj = objDict['function']
            if 'jacFunction' in objDict:
                self.jac = objDict['jacFunction']
            if 'hessFunction' in objDict:
                self.hess = objDict['hessFunction']
        if self.constraints:
            sol = optimize.minimize(self.obj, x0, method=self.method,jac=self.jac,hess=self.hess,constraints=self.constraints,options={'disp': False})
        else:
            sol = optimize.minimize(self.obj, x0, method=self.method,jac=self.jac,hess=self.hess,options={'disp': False})
        return sol