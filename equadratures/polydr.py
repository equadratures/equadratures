#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:15:28 2017

@author: Henry S. Yuchi
"""


from equadratures import *
import scipy.stats as stats
import numpy as np

order = 2
x1 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x2 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x3 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x4 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
x5 = Parameter(param_type='Uniform',  lower=-1.0,  upper=1.0,  points=order+1)
#x5 = Parameter(param_type='Gaussian', shape_parameter_A=0, shape_parameter_B=1, points=order+1)
parameters = [x1, x2, x3, x4, x5]

basis = IndexSet('Total order',  [order,  order,  order,  order,  order] )
maximum_number_of_evals = basis.cardinality

def fun(x):
    #A=np.random.rand(5, 5)
    A=np.diag([0.9, 0.7, 0.5, 0.3, 0.1])
    a=np.dot(A, x)
    b=np.dot(a, x)
    return 0.5*b
    
poly = Polylsq(parameters, basis)
poly.set_no_of_evals(maximum_number_of_evals)
coefficients_computed = poly.computeCoefficients(fun)
pts = np.mat(np.random.rand(3, 5)) + 1
vv = poly.getPolynomialApproximation(pts, fun)
#print vv

A=np.diag([0.9, 0.7, 0.5, 0.3, 0.1])
B=np.dot(A, A)/3
    
#Now start eigendecomposition for dimension reduction
u, s, v=np.linalg.svd(B)
print s
print u



def polydr(indexset, parameters, finiteflag=1):
    #indexset points to the indices, parameters correspond to the parameter
    #settings, finiteflag refers to whether finite difference is used
    
    #N notes number of variables
    N=len(parameters)
    if finiteflag==1:
        #specify number of samples
        num=200
        #store the min-max values for samples in each dimension
        minmax=np.zeros((2,N))
        #store the sample coordinates
        samples=np.zeros((num,N))
        #store the probabilities of each sample
        prob=np.ones(num)/(num*1.0)
        #calculate the gradients and probabilities by finite difference. Store
        # the min-max values in each dimension
        for i in range(0,N):
            if parameters[i].param_type is 'Uniform':
                minmax[0,i]=parameters[i].lower
                minmax[1,i]=parameters[i].upper
                #generate samples
                samples[:,i],comp=uniformsampling(minmax[0,i],minmax[1,i],num)
                
            elif parameters[i].param_type is 'Gaussian':
                #generate samples
                samples[:,i],comp=gaussiansampling(parameters[i].shape_parameter_A,parameters[i].shape_parameter_B,num)
                #from the samples calculate the min-max
                minmax[0,i]=samples[:,i].min()
                minmax[1,i]=samples[:,i].max()
                for j in range(0,num):
                    prob[j]=prob[j]*comp[j]
                prob=prob/sum(prob)
        #now that all samples with probabilities are drawn, calculate the 
        #gradients
        print sum(prob)
        gradinterval=np.zeros(N)
        for i in range(0, N):
            gradinterval[i]=(minmax[1,i]-minmax[0,i])/50000.0
        C=np.zeros((N,N))
        for i in range(0,num):
            grad=getgrad(gradinterval, samples[i])
            C=C+np.outer(grad, grad)*prob[i]
        #eigendecomposition to find eigenvectors and eigenvalues
        u, s, v=np.linalg.svd(C)
        return s,u,C
    elif finiteflag==0:
        #Still left to be filled in...
        return 0,0,0
        
        
    
def uniformsampling(min,max,num):
    out=np.random.uniform(min,max,num)
    prob_comp=np.ones(num)/num
    return out,prob_comp

def gaussiansampling(mean,deviation,num):
    out=np.random.normal(mean,deviation,num)
    prob_comp=stats.norm.pdf(out,mean,deviation)
    prob_comp=prob_comp/sum(prob_comp)
    return out, prob_comp
    
def getgrad(gradinterval,coords):
    N=gradinterval.shape[0]
    gradient=np.zeros(N)
    for i in range(0,N):
        coord1=np.zeros(N)
        coord2=np.zeros(N)
        for j in range(0,N):
            coord1[j]=coords[j]
            coord2[j]=coords[j]
        coord1[i]=coord1[i]+gradinterval[i]
        coord2[i]=coord2[i]-gradinterval[i]
        output1=estimate(coord1)
        output2=estimate(coord2)
        gradient[i]=(output1-output2)/(2*gradinterval[i])
    return gradient

def estimate(coords):
    coords=np.mat(coords)
    #this function returns the estimated value of the polynomial
    g = poly.getPolynomialApproximation(coords,  fun)
    return g


    
    
    
eigenvalue,eigenvector,  C=polydr(basis.elements, parameters)
print eigenvalue
print eigenvector    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    