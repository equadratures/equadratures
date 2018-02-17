from equadratures import *
import numpy as np
import scipy as sp

## Basic test
# Example l1eq_example.m
#N = 512
#T = 20
#K = 120
#
#results = np.zeros(10)
##np.random.seed(seed=1)
#for i in range(10):
#    x = np.zeros(N)
#    
#    #Random init
#    q = np.random.permutation(N)
#    x[q[:T]] = np.sign(np.random.randn((T)))
#    A = np.random.randn(K,N)
#    A = sp.linalg.orth(A.T).T
#    y = np.dot(A,x).flatten()
##    y = y + .005*np.random.normal(size = y.shape)
#    
#    #Set init
#    #A = np.zeros((K,N))
#    #f = open('x.txt','r')
#    #for i in range(512):
#    #    x[i] = f.readline()
#    #f.close()    
#    #g = open('A.txt','r')
#    #for i in range(120):
#    #    A[i,:] = g.readline()[:-1].split(',')
#    #g.close()
#    #h = open('y.txt','r')
#    #y = np.zeros(K)
#    #for i in range(120):
#    #    y[i] = h.readline()
#    #h.close()
#    
#    
#    
#    
#    epsilon = .005 * np.sqrt(K) * np.sqrt(1.0 + 2.0*np.sqrt(2.0) / np.sqrt(K))
#    
#    #x0 = np.dot(A.T,y)
#    
##    xp = bp_denoise(A,y,epsilon, verbose = False)
#    xp = bp(A,y)
#    print np.linalg.norm(x.flatten()-xp)/np.linalg.norm(x)
#    results[i] = np.linalg.norm(x.flatten()-xp)/np.linalg.norm(x)
#print "total mean"
#print np.mean(results)

## Poly tests
def fun(x):
    return np.exp(x[0])

p_order = 5

x0 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x1 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x2 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x3 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)

parameters = [x0,x1,x2,x3]

orders = [p_order,p_order,p_order,p_order]
polybasis = IndexSet("Total order", orders)
success = 0.0
#for i in range(100):
#x_cs = np.random.uniform(size = (int(polybasis.elements.shape[0]/2),len(orders)))
x_reg = np.random.uniform(size = (1000,len(orders)))

#polycs = Polycs( parameters, polybasis,training_x = x_cs, fun=fun)
polycs = Polycs( parameters, polybasis,sampling = "dlm", fun=fun)
polyreg = Polyreg(x_reg, parameters, polybasis, fun=fun)

coeffs_cs = polycs.coefficients
coeffs_reg = polyreg.coefficients
print np.linalg.norm(coeffs_cs - coeffs_reg)/np.linalg.norm(coeffs_reg)
#p,w = polycs.getQuadratureRule(options = "qmc")
#print len(coeffs_reg[coeffs_reg>1e-5])
#    if np.linalg.norm(coeffs_cs - coeffs_reg)/np.linalg.norm(coeffs_reg) <= 0.001:
#        success += 0.01
#print polycs.sample_X(parameters, polybasis, "asymptotic", 3)        
#print success