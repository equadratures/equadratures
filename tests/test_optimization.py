import numpy as np
from scipy.linalg import orth
from pyDOE import lhs
import equadratures as eq
from optimization import Optimization


def ObjFun(x,a):
    v = np.dot(x,a)
    f = np.zeros((v.shape[0]))
    c = 5
    for i in range(f.shape[0]):
        for j in range(v.shape[1]):
            f[i] += 0.5*((c*v[i,j])**4 - 16*(c*v[i,j])**2 + 5*c*v[i,j])
    return v,f    
    
def ConFun(x,a):
    v = np.dot(x,a)
    g = np.zeros((v.shape[0]))
    for i in range(g.shape[0]):
        for j in range(v.shape[1]):
            g[i] += v[i,j]**2
    return v, g  

if __name__ == '__main__':  
    
    degf = 4
    degg = 2
    df = 2
    dg = 2 
    plotlim = 6
    boundsg = [2.0,3.0]
    
    n = 30
    
    NTotal = 5000
    Ntrain = int(0.8*NTotal)
    Ntest = NTotal-Ntrain
    
    xtrain = -1.+2.*lhs(n, samples=Ntrain)
    xtest = -1.+2.*lhs(n, samples=Ntest)
            
#   Real active subspace and values for f
    U = orth(np.random.rand(n,df))
    utrain,ftrain = ObjFun(xtrain,U)
    utest,ftest = ObjFun(xtest,U)
    
#   Real active subspace and values for g2
    W = orth(np.random.rand(n,dg))
    wtrain,gtrain = ConFun(xtrain,W)
    wtest,gtest = ConFun(xtest,W)
    
    fparam = eq.Parameter(distribution='uniform', lower=-6, upper=6., order=degf)
    gparam = eq.Parameter(distribution='uniform', lower=-6, upper=6., order=degg)
    fParameters = [fparam for i in range(df)]
    gParameters = [gparam for i in range(dg)]
    myBasis = eq.Basis('Total order')
    fpoly = eq.Polyreg(fParameters, myBasis, training_inputs=utrain, training_outputs=ftrain)
    gpoly = eq.Polyreg(fParameters, myBasis, training_inputs=wtrain, training_outputs=gtrain)
    
    Opt = Optimization(method='SLSQP')
    Opt.AddObjective({'poly': fpoly,'U':U})
    Opt.AddNonLinearIneqCon({'poly':gpoly,'bounds':boundsg,'U':W})
    Opt.AddLinearIneqCon(np.eye(n),-np.ones(n),np.ones(n))
    x0 = np.random.rand(n)
    print Opt.OptimizePoly(x0)