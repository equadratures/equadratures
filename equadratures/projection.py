from .optimization import Optimization
import numpy as np
from scipy.spatial import ConvexHull

class Projection:
    def __init__(self,poly,subspace,bounds):
        self.opt = None
        self.poly = poly
        self.subspace = subspace
        self.bounds = bounds
    @staticmethod
    def hullProjection(V,X):
        n = V.shape[1]
        convexHull = ConvexHull(V)
        A = convexHull.equations[:,:n]
        b = -convexHull.equations[:,n]
        vertV = V[convexHull.vertices,:]
        vertX = X[convexHull.vertices,:]
        return {'A': A, 'b': b, 'vertV': vertV, 'vertX': vertX}
    def defineInequalityCons(self):
        n = self.subspace.shape[0]
        opt = Optimization(method='trust-constr') 
        opt.addLinearIneqCon(np.eye(n),-np.ones(n),np.ones(n))
        opt.addNonLinearIneqCon({'poly':self.poly,'bounds':self.bounds,'subspace':self.subspace})
        self.opt = opt
        return None
#    def addDistCons(self,v,dist):
#        function = lambda x: x.dot(self.subspace).dot(self.subspace.T).dot(x) - v.dot(self.subspace.T).dot(x) - x.dot(self.subspace).dot(v) + v.dot(v)
#        jacFunction = lambda x: 2.0*self.subspace.dot(self.subspace.T).dot(x) - v.dot(self.subspace.T)
#        hessFunction = lambda x,v: 2.0*self.subspace.dot(self.subspace.T)
#        conDict = {'function':function,'jacFunction':jacFunction,'hessFunction':hessFunction,'bounds':[0.0,dist]}
#        self.opt.addNonLinearIneqCon(conDict)
#        return None
    def maxDirectionOpt(self,direction,U):
        n = U.shape[0]
        c = U.dot(direction)
        x0 = np.random.uniform(-1,1,n)
        objDict = {'function': lambda x: c.dot(x), 'jacFunction': lambda x: c, 'hessFunction': lambda x: np.zeros((n,n))}
        x = self.opt.optimizePoly(objDict,x0)['x']
        return x
    def setProjection(self,W):
        self.defineInequalityCons()
        n, d = W.shape
        X = np.zeros((1,n))
        OK = 0
        MaxIter = 5
        cnt = 0
        
        while not OK:
            direction = np.random.uniform(-1,1,d)
            if cnt > MaxIter:
                raise Exception('Iterative hull algorithm exceeded maximum number of iterations.')
            x = self.maxDirectionOpt(direction,W)
            cnt += 1
            X = np.vstack((X,x))
            V = np.dot(X,W)
            V,ind = np.unique(V.round(decimals=4),return_index=True,axis=0)
            X = X[ind]
            if V.shape[0] == d+2:
                OK = 1
        X = X[~np.all(X == 0., axis=1)]
        V = V[~np.all(V == 0., axis=1)]
        
        P1 = self.hullProjection(V,X)
        OK = 0
        while not OK:
            for i in range(P1['A'].shape[0]):
                if cnt > MaxIter:
                    print 'Exceeded number of maximum number of iterations'
                    return P1
                direction = P1['A'][i,:]
                x = self.maxDirectionOpt(direction,W)
                cnt += 1
                v = np.dot(x,W)
#                self.addDistCons(v,0.01)
                X = np.vstack((X,x))
                V = np.vstack((V,v))
                V,ind = np.unique(V.round(decimals=4),return_index=True,axis=0)
                X = X[ind]
            P2 = self.hullProjection(V,X)
            if P1['vertV'].shape == P2['vertV'].shape:
                if np.allclose(P1['vertV'],P2['vertV']):
                    OK = 1
            P1 = P2
        return P1