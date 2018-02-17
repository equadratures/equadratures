import numpy as np 


def activeSubspaces(Poly, X=None):
    
    # Any theoretical rationale for selecting number of samples?
    # Shouldn't this be some log scale factor!?
    M = 200
    d = Poly.dimensions
    if X is  None:
        X = np.zeros((M, d))
        for j in range(0, d):
            X[:, j] = np.reshape( Poly.parameters[j].getSamples(M) , (M, 1) )
    polygrad = Poly.getPolynomialGradientApproximant(X)
    # Construct covariance matrix!
    C = np.zeros((d, d))
    for i in range(0, d):
        grad_f = np.mat( np.reshape(polygrad[:,i], (d, 1)) )
        C =+ grad_f * grad_f.T 
    C = 1./float(M) * C 

    # Compute eigendecomposition!   
    e, W = np.linalg.eigh(C)

def activeSubspacesFD(Poly, X=None, h=None):
    if h is None:
        h = 1e-6 # perhaps need to change this...    
    
    radinterval=np.zeros(N)
        for i in range(0, N):
            gradinterval[i]=(minmax[1,i]-minmax[0,i])/50000.0
        C=np.zeros((N,N))
        for i in range(0,num):
            grad=getgrad(gradinterval, samples[i])
            C=C+np.outer(grad, grad)*prob[i]
