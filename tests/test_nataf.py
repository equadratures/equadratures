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
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = -8.0, shape_parameter_B=1.9))
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 12.9, shape_parameter_B = 3.9))

#R = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.7], [0.5, 0.7, 1.0]])
R = np.identity(len(D))
for i in range(len(D)): 
    for j in range(len(D)):
        if i==j:
            continue
        else:
            R[i,j] = 0.8
#print R

obj = Nataf(D,R)
#xu = obj.U2C(R)
#print xu 
""" testing the passage through the inputs
"""
o = obj.getCorrelatedSamples(5000)
#print o
# plot of results:

t  = o[0]# first matrix: input: uncorrelated
tt = o[1]# second matrix: output: correlated.
#print t

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.grid(linewidth=0.4, color='k')
plt.plot(t[:,0], t[:,1],'ro', label='w/o correlations')
plt.plot(tt[:,0], tt[:,1], 'bo', label='w/ correlations')
plt.legend(loc='upper right')
adjust_spines(ax, ['left', 'bottom'])
plt.title('Simple Cholesky Trick')
plt.show()

print 'mean of uncorrelated inputs', np.mean(t[:,0]) , np.mean(t[:,1]) 
print 'mean of correlated inputs', np.mean(tt[:,0]) , np.mean(tt[:,1]) 
print 'variance of uncorrelated inputs', np.var(t[:,0]) , np.var(t[:,1]) 
print 'variance of correlated inputs', np.var(tt[:,0]) , np.var(tt[:,1]) 
#------------------------------------------------------#
# testing transformations: direct

u = obj.C2U(tt)
#print u
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.grid(linewidth=0.4, color='k')
plt.plot(tt[:,0], tt[:,1], 'bo', label='Correlated inputs')
plt.plot(u[:,0], u[:,1],'ro', label='Uncorrelated outputs')
plt.legend(loc='upper right')
plt.title('Nataf transformation')
adjust_spines(ax, ['left', 'bottom'])
plt.show()


#------------------------------------------------------#
# testing the mean and the variance of output marginals
for i in range(len(D)):
    print 'mean of ',i,'th output:', np.mean(u[:,i])
for i in range(len(D)):
    print 'variance of ',i,'th output:', np.var(u[:,i])


"""
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
adjust_spines(ax, ['left', 'bottom'])
plt.title('inverse Nataf: check the inverse mapping')
plt.show()

#------------------------------------------------------#
# testing the mean and the variance of output marginals

print '-----------------------------------------------'
for i in range(len(D)):
    print 'mean of ',i,'th output:', np.mean(u[:,i])
for i in range(len(D)):
    print 'variance of ',i,'th output:', np.var(u[:,i])
print '-----------------------------------------------'

"""