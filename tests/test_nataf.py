import numpy as np
from equadratures import *
import matplotlib.pyplot as plt

#---------------------------------------------------------------#
#       list of objects==distributions from parameter
#               testing instance to parameter
#---------------------------------------------------------------#

mean1 = 0.4
var1  = 1.3
low1   = 0.2
up1    = 1.15

mean2 = 0.7
var2  = 3.0
low2   = 0.3
up2    = 0.5

D = list()
D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=1.0))
D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=4.0))

#D.append(Parameter(order=3, distribution='uniform', lower=0.5, upper=0.8))
#print 'from test_ len D:', len(D)
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 10.0, shape_parameter_B=16.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))

#D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.4, shape_parameter_B = 2.8))
#D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 3.2, shape_parameter_B = 1.5))

#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 10.0, shape_parameter_B=16.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 10.0, shape_parameter_B=16.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 10.0, shape_parameter_B=16.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 30.0, shape_parameter_B = 4.0))
#D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
#D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
#D.append(Parameter(order=3, distribution='Cauchy', shape_parameter_A = 0.5, shape_parameter_B = 0.7))
#D.append(Parameter(order=3, distribution='Cauchy', shape_parameter_A = 0.5, shape_parameter_B = 0.7))
#D.append(Parameter(order=3, distribution='Chebyshev', upper=1.0, lower=0.0))
#D.append(Parameter(order=3, distribution='Chebyshev', upper=1.0, lower=0.0))
#D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=1.5))
#D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=1.5))
#D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
#D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
#D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 0.7, shape_parameter_B = 0.8))
#D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 0.7, shape_parameter_B = 0.8))
#D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
#D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
#D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 0.9, shape_parameter_B =4.8, upper = 3.0, lower = 8.9))
#D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 0.9, shape_parameter_B =4.8, upper = 3.0, lower = 8.9))
#D.append(Parameter(order=3, distribution='weibull', shape_parameter_A=0.8, shape_parameter_B=0.9))
#D.append(Parameter(order=3, distribution='weibull', shape_parameter_A=0.8, shape_parameter_B=0.9))


#----------------------------------------------------------------------#
R = np.identity(len(D))
for i in range(len(D)): 
    for j in range(len(D)):
        if i==j:
            continue
        else:
            R[i,j] = 0.60
#print R

obj = Nataf(D,R)
#xu = obj.U2C(R)
#print xu 
""" testing the passage through the inputs
"""
o = obj.getCorrelatedSamples(N=1000)
#print o
# plot of results:

t  = o[0]# first matrix: input: uncorrelated
tt = o[1]# second matrix: output: correlated.
#print t


#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#plt.grid(linewidth=0.4, color='k')
#plt.plot(t[:,0], t[:,1],'ro', label='w/o correlations')
#plt.plot(tt[:,0], tt[:,1], 'bo', label='w/ correlations')
#plt.legend(loc='upper right')
#adjust_spines(ax, ['left', 'bottom'])
#plt.title('Simple Cholesky Trick')
#plt.xlim(0,20)
#plt.ylim(24,44)
#plt.show()

#print 'mean of uncorrelated inputs', np.mean(t[:,0]) , np.mean(t[:,1]) 
print 'mean of uncorrelated input: FROM OBJECT', obj.D[0].mean, obj.D[1].mean
print 'mean of uncorrelated input: AFTER PASSAGE INTO METHOD:', np.mean(t[:,0]), np.mean(t[:,1])
print 'mean of correlated outputs', np.mean(tt[:,0]) , np.mean(tt[:,1]) 
#print 'variance of uncorrelated inputs', np.var(t[:,0]) , np.var(t[:,1]) 
print 'variance of uncorrelated inputs: FROM OBJECT', obj.D[0].variance, obj.D[1].variance 
print 'variance of uncorrelated input: AFTER PASSAGE INTO METHOD:', np.var(t[:,0]), np.var(t[:,1])
print 'variance of correlated outputs', np.var(tt[:,0]) , np.var(tt[:,1]) 
#------------------------------------------------------#
# testing transformations: direct

u = obj.C2U(tt)
#print u
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.grid(linewidth=0.4, color='k')
plt.plot(u[:,0], u[:,1],'ro', label='Uncorrelated outputs')
plt.plot(tt[:,0], tt[:,1], 'bo', label='Correlated inputs')
plt.legend(loc='upper right')
plt.title('Nataf transformation')
plt.xlim(-10,40)
plt.ylim(-10,40)
adjust_spines(ax, ['left', 'bottom'])
plt.show()


#------------------------------------------------------#
# testing the mean and the variance of output marginals
print 'direct transformation:'
for i in range(len(D)):
    print 'mean of ',i,'output:', np.mean(u[:,i])
for i in range(len(D)):
    print 'variance of ',i,'output:', np.var(u[:,i])



#------------------------------------------------------#
# testing transformations: inverse

c = obj.U2C(u)
#print u
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.grid(linewidth=0.4, color='k')
plt.plot(u[:,0], u[:,1], 'bo', label='uncorrelated in')
plt.plot(c[:,0], c[:,1],'ro', label='correlated out')
plt.legend(loc='upper right')
plt.xlim(-5,40)
plt.ylim(-5,40)
adjust_spines(ax, ['left', 'bottom'])
plt.title('inverse Nataf: check the inverse mapping')
plt.show()

#------------------------------------------------------#
# testing the mean and the variance of output marginals

print '-----------------------------------------------'
print 'inverse transformation:'
for i in range(len(D)):
    print 'mean of ',i,'th output:', np.mean(c[:,i])
for i in range(len(D)):
    print 'variance of ',i,'th output:', np.var(c[:,i])
print '-----------------------------------------------'


