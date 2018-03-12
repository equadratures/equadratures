"""Dimension Reduction Functionalities"""
import numpy as np

import scipy
from parameter import Parameter
from poly import Poly
import scipy.io
from basis import Basis


def computeActiveSubspaces(PolynomialObject, samples=None):
    d = PolynomialObject.dimensions
    if samples is  None:
        M = 300 # Replace with log factor x d
        X = np.zeros((M, d))
        for j in range(0, d):
            X[:, j] =  np.reshape(PolynomialObject.parameters[j].getSamples(M), M)
    else:
        X = samples
        M, _ = X.shape
        X = samples

    # Gradient matrix!
    polygrad = PolynomialObject.evaluatePolyGradFit(xvalue=X)
    weights = np.ones((M, 1)) / M
    R = polygrad.transpose() * weights
    C = np.dot(polygrad, R )

    # Compute eigendecomposition!
    e, W = np.linalg.eigh(C)
    idx = e.argsort()[::-1]
    eigs = e[idx]
    eigVecs = W[:, idx]
    return eigs, eigVecs


def linearModel(Xtrain, ytrain,bounds):
    #INPUTS
    #Xtrain and ytrain are regarded as numpy array
    #Xtrain is N*D, ytrain is N*1, bounds is D*2
    #OUTPUS
    #u is the coefficients for each X sample D*1 array
    #c is the bias, a number

    #first standardization
    N,D=Xtrain.shape
    X_stnd=np.zeros((N,D))
    for i in range(0,N):
        for j in range(0,D):
            X_stnd[i,j]=2*(Xtrain[i,j]-bounds[j,0])/(bounds[j,1]-bounds[j,0])-1
    #then start the ordinary least squares formulation
    A=np.concatenate((X_stnd,np.ones((N,1))),axis=1)
    x=np.linalg.lstsq(A,ytrain)[0]
    u=x[0:D-1]
    c=x[D]
    return u,c

def standard(X,bounds):
    #INPUTS X array M*m
    #INPUTS bounds array m*2
    #OUTPUTS X_stnd array M*m
    M,m=X.shape
    X_stnd=np.zeros((M,m))
    for i in range(0,M):
        for j in range(0,m):
            X_stnd[i,j]=2*(X[i,j]-bounds[j,0])/(bounds[j,1]-bounds[j,0])-1
    return X_stnd

def vandermonde(eta,p):
    #this function establishes the Vandermonde matrix
    #this is part of the variable projection algorithm
    #INPUTS
    #eta M*n array
    #p constant
    #OUTPUTS
    #V array M*N (N is calculated)

    _,n=eta.shape
    listing=[]#the list to store the polynomial degrees
    for i in range(0,n):
        listing.append(p)
    Object=Basis('Total order',listing)
    #Establish n Parameter objects
    params=[]
    P=Parameter(order=p,lower=-1,upper=1,param_type='Uniform')
    for i in range(0,n):
        params.append(P)
    #Then use the params list to establish the Poly object
    Polybasis=Poly(params,Object)
    V=Polybasis.getPolynomial(eta)
    V=V.T
    return V,Polybasis

def jacobian(V,V_plus,U,y,f,Polybasis,eta,minmax,X):
    #this function genrates the jacobian tensor using derivation from Jeff's
    #variable projection algorithm
    #INPUTS
    #V array M*N
    #V_plus array N*M
    #U array m*n
    #y array M*n
    #f array M*1
    #Polybasis Object for the polynomial basis
    #eta array M*n
    #minmax array n*2
    #X array M*m
    #OUTPUT
    #J tensor M*m*n
    M,N=V.shape
    m,n=U.shape
    Gradient=Polybasis.getPolynomialGradient(eta)#a list of gradient matrices
    #n*1 matrices contained, each is an array M*N
    #Use the minmax matrix to obtain a and d
    #a array n*1
    #d array n*1
    #eta=a+diag(d)y y array M*n
    sub=minmax[:,1]-minmax[:,0]# n*1 array
    vectord=np.reshape(2.0/sub,(2,1))
    #Initialize the tensor
    J=np.zeros((M,m,n))
    #Then obtain the derivative of this tensor
    dV=np.zeros((m,n,M,N))
    for k in range(0,m):
        for l in range(0,n):
            for i in range(0,M):
                for j in range(0,N):
                    current=Gradient[l].T
                    dV[k,l,i,j]=np.asscalar(vectord[l])*np.asscalar(X[i,k])*np.asscalar(current[i,j])

    #first get the P matrix
    P=np.identity(M)-np.matmul(V,V_plus)#M*M array
    V_minus=scipy.linalg.pinv(V)#N*M array
    #Then fill in the elements in the tensor
    for j in range(0,m):
        for k in range(0,n):
            temp1=np.linalg.multi_dot([P,dV[j,k,:,:],V_minus])
            J[:,j,k]=(-np.matmul((temp1+temp1.T),f)).reshape((M,))
    return J


def variable_projection(X,f,n,p,gamma,beta):
    #INPUTS
    #sample inputs X array M*m
    #function values f array M*1
    #subspace dimension n
    #polynomial degree p
    #step length reduction factor gamma \in (0,1)
    #Armijo tolerance beta \in (0,1)

    #OUTPUTS
    #active subspace U array M*n
    #polynomial coeffcients c array N*1 (N is calculated in the function)

    #assume uniform sampling for now
    M,m=X.shape
    Z=np.random.rand(m,n)*2-1
    U,R=np.linalg.qr(Z)

    #set a convergence flag
    convflag=0

    while convflag==0:
        y=np.dot(X,U)# an array M*n
        #then scaling to ensure y \in (-1 1)^n
        minmax=np.zeros((n,2))
        for i in range(0,n):
            minmax[0,i]=min(y[:,i])
            minmax[1,i]=max(y[:,i])
        #now given the bounds we can do standardization
        eta=np.zeros((M,n))#\eta is the affine transformation of y, M*n
        for i in range(0,M):
            for j in range(0,n):
                eta[i,j]=2*(y[i,j]-minmax[0,j])/(minmax[1,j]-minmax[0,j])-1
        convflag=1
        #now establish the Legendre basis
        #The Vandermonde matrix contains up to p-degree of polynomials and n
        #degree of input subspaces
        V,Polybasis=vandermonde(eta,p)# V is array M*N
        #now get the psuedoinverse
        V_plus=np.linalg.pinv(V)#array N*M
        #now get the coefficients
        coeff=np.dot(V_plus,f)
        #calculate the residual
        res=f-np.dot(V,coeff)
        #now build the Jacobian

        J=jacobian(V,V_plus,U,y,f,Polybasis,eta,minmax,X)

        #get the gradient
        G=np.zeros((m,n))
        for i in range(0,M):
            G=G+res[i]*J[i,:,:]
        #conduct the SVD for J_vec
        vec_J=np.zeros((M,(m*n)))
        for i in range(0,M):
            for j in range(0,m):
                for k in range(0,n):
                    vec_J[i,j*n+k]=J[i,j,k]
        Y,S,Z=np.linalg.svd(vec_J,full_matrices=False)
        #carry out Gauss-Newton step
        vec_delta=np.zeros(((m*n),1))
        temp=np.dot(Y.T,res)
        for i in range(0,(m*n-n*n)):
            vec_delta+=temp[i]*np.reshape(Z[:,i],(m*n,1))/S[i]
        delta=np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                delta[i,j]=vec_delta[i*n+j]
        #vectorize G
        vec_G=np.zeros(((m*n),1))
        for i in range(0,m):
            for j in range(0,n):
                vec_G[i*n+j]=G[i,j]
        alpha=np.matmul(vec_G.T,vec_delta)[0,0]
        #check alphd
        if alpha>=0:
            delta=G
            vec_delta=np.zeros(((m*n),1))
            for i in range(0,m):
                for j in range(0,n):
                    vec_delta[i*n+j]=delta[i,j]
            alpha=np.matmul(vec_G.T,vec_delta)[0,0]

        #SVD on delta
        Y,S,Z=np.linalg.svd(delta,full_matrices=False)
        t=1/gamma
        #flag to break the bakctracking line search
        breakflag=0
        while breakflag==0:
            t=t*gamma
            print t
            U_plus=np.linalg.multi_dot([U,Z,np.cos(np.diag(S)*t),Z.T])+np.linalg.multi_dot([Y,np.sin(np.diag(S)*t),Z.T])
            #Update the values with the new U matrix
            y=np.dot(X,U_plus)# an array M*n
            #then scaling to ensure y \in (-1 1)^n
            minmax=np.zeros((n,2))
            for i in range(0,n):
                minmax[0,i]=min(y[:,i])
                minmax[1,i]=max(y[:,i])
            #now given the bounds we can do standardization
            eta=np.zeros((M,n))#\eta is the affine transformation of y, M*n
            for i in range(0,M):
                for j in range(0,n):
                    eta[i,j]=2*(y[i,j]-minmax[0,j])/(minmax[1,j]-minmax[0,j])-1
            V,Polybasis=vandermonde(eta,p)
            res_plus=f-np.linalg.multi_dot([V,np.linalg.pinv(V),f])
            if np.linalg.norm(res_plus)<=np.linalg.norm(res)+alpha*beta*t or t<1e-10:
                breakflag=1
        #Now check convergence of U
        diff=np.absolute(U-U_plus)#here diff is a m*n matrix
        if diff.max()<0.002:
            convflag=1
        U=U_plus
    return U



docu=scipy.io.loadmat('Data.mat')
x_100=docu['x_100']
eff_100=docu['eff_100']
bounds=np.concatenate((np.matlib.repmat(np.array([-0.2,0.2]),15,1),
                          np.matlib.repmat(np.array([-0.1,0.1]),10,1)),axis=0)
X=standard(x_100,bounds)
#first check the linear model
u,c=linearModel(x_100,eff_100,bounds)
#then run the variable projection
U=variable_projection(X,eff_100,2,2,0.05,1e-8)
print U

