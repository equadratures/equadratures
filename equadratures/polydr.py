#!/usr/bin/python
from equadratures import *
import numpy as np
from scipy.stats import *

order = 2
x1 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x2 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x3 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x4 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x5 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
parameters = [x1, x2, x3, x4, x5]

basis = IndexSet('Total order',  [order,  order,  order,  order,  order] )
maximum_number_of_evals = basis.cardinality

def fun(x):
    A=np.random.rand(5, 5)
    #A=diag([0.9, 0.7, 0.5, 0.3, 0.1])
    a=np.dot(A, x)
    b=np.dot(a, x)
    return 0.5*b
    
poly = Polylsq(parameters, basis)
poly.set_no_of_evals(maximum_number_of_evals)
coefficients_computed = poly.computeCoefficients(fun)
pts = np.mat(np.random.rand(3, 5)) + 1
vv = poly.getPolynomialApproximation(pts, fun)
print vv

# 1. Compare truth with approximation!
# 2. DR and write down analytical solution!

def polydr(indexset, pts, grad=None):
    dimension=indexset.shape[1]
    minmax=np.zeros((2, dimension))
    
    for i in range(0, dimension):
        #minmax[0][i]=pts[:, i].min()-0.01
        #minmax[1][i]=pts[:, i].max()+0.01
        minmax[0, i]=-1
        minmax[1, i]=1
    gradinterv=np.zeros(dimension)
    
    for i in range(0, dimension):
        gradinterv[i]=(minmax[1][i]-minmax[0][i])/10000.0
    numofsample=100;
    
    samplecoords, prob=uniformsampling(minmax, numofsample)
    #samplecoords, prob=gaussiansampling(minmax, numofsample)
    
    C=np.zeros((dimension, dimension))
    for i in range(0, numofsample):
        grad=getgrad(gradinterv, samplecoords[i])
        C=C+np.outer(grad, grad)*prob[i]
    
    
    #Now start eigendecomposition for dimension reduction
    u, s, v=np.linalg.svd(C)
    
    # here s is the array for all eigenvalues N*1
    # u is the array for all eigenvectors N*N
    t=range(1, dimension+1)
    #plt.plot(t, s, 'ro')
    #plt.show()

    
    A=np.diag([0.9, 0.7, 0.5, 0.3, 0.1])
    B=np.dot(A, A)/3
    
    #Now start eigendecomposition for dimension reduction
    u, s, v=np.linalg.svd(B)
    
    # here s is the array for all eigenvalues N*1
    # u is the array for all eigenvectors N*N
    t=range(1, dimension+1)
    #plt.plot(t, s, 'ro')
    #plt.show()

    
    
    return s, u, C
    
    
def getgrad(gradinterv, coords):
    #the point of this function is to calculate the gradient vector at the point specified by the coords
    #coords is supposed to be a N*1 vector
    #df=f(x+h)-f(x-h)/2h
    N=coords.shape[0]
    grad=np.zeros(N)
    for i in range(0, N):
        newcoord1=np.zeros(N)
        newcoord2=np.zeros(N)
        for j in range(0, N):
            newcoord1[j]=coords[j]
            newcoord2[j]=coords[j]
        newcoord1[i]=newcoord1[i]+gradinterv[i]
        newcoord2[i]=newcoord2[i]-gradinterv[i]
        est1=est(newcoord1)
        est2=est(newcoord2)
        grad[i]=(est2-est1)/(2*gradinterv[i])
        #print newcoord1, newcoord2, coords
    return grad

def est(coords):
    coords=np.mat(coords)
    #this function returns the estimated value of the polynomial
    g = poly.getPolynomialApproximation(coords,  fun)
    return g
    
def uniformsampling(minmax, num):
    N=minmax.shape[1]
    samplecoords=np.zeros((num, N))
    prob=np.ones(num)/num
    for i in range(0, N):
        samplecoords[:, i]=np.random.uniform(minmax[0][i], minmax[1][i], num)
    return samplecoords, prob

def gaussiansampling(minmax, num):
    #take the minmax as 3sigma for gaussian sampling in a multivariate gaussian normal
    N=minmax.shape[1]
    samplecoords=np.zeros((num, N))
    prob=np.zeros(num)
    for i in range(0, N):
        samplecoords[:, i]=np.random.uniform((minmax[0][i]+minmax[1][i])/2.0, (minmax[1][i]-minmax[0][i])/6.0, num)
    #now construct the mean and cov matrix for prob calculation
    mean=(minmax[0]+minmax[1])/2.0
    cov=np.zeros((N, N))
    for i in range(0, N):
        cov[i, i]=((minmax[1, i]-minmax[0, i])/6.0)**2
    prob=np.multivariate_normal.pdf(samplecoords,  mean, cov)
    prob=prob*1.0/sum(prob)
    return samplecoords, prob
    
eigenvalue,eigenvector,  C=polydr(basis.elements, pts)
   
