import numpy as np
from equadratures import *
from natafclass import *
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
#D.append(Parameter(order=3, distribution='truncated-gaussian',lower=low1, upper=up1, shape_parameter_A = mean1, shape_parameter_B = var1))
#D.append(Parameter(order=3, distribution='truncated-gaussian',lower=low2, upper=up2, shape_parameter_A = mean2, shape_parameter_B = var2))
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 1.0, shape_parameter_B=1.9, lower=-5.0, upper =5.0))
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 2.9, shape_parameter_B = 3.9, lower = -10.0, upper = 10.0))

#R = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.7], [0.5, 0.7, 1.0]])
R = np.identity(len(D))
for i in range(len(D)): 
    for j in range(len(D)):
        if i==j:
            continue
        else:
            R[i,j] = 0.8
#print R

obj = Natafclass(D,R)
#xu = obj.U2C(R)
#print xu 
""" testing the passage through the inputs
"""
o = obj.getCSamples(400)
#print o
# plot of results:

t  = o[0]# first matrix: input: uncorrelated
tt = o[1]# second matrix: output: correlated.
#print t

plt.figure()
plt.grid(linewidth=0.4, color='k')
plt.plot(t[:,0], t[:,1],'ro', label='uncorrelated in')
plt.plot(tt[:,0], tt[:,1], 'bo', label='correlated out')
plt.legend(loc='upper right')
plt.title('get correlated, N input')
plt.xlim(-10,15)
plt.ylim(-10,15)
plt.show()

oo = obj.getCSamples(points=t)
#print oo
t  = oo[0]# first matrix: input: uncorrelated
tt = oo[1]# second matrix: output: correlated.

plt.figure()
plt.grid(linewidth=0.4, color='k')
plt.plot(t[:,0], t[:,1],'ro', label='uncorrelated in')
plt.plot(tt[:,0], tt[:,1], 'bo', label='correlated out')
plt.legend(loc='upper right')
plt.title('get correlated, points input')
plt.xlim(-10,15)
plt.ylim(-10,15)
plt.show()

ooo = obj.getUSamples(300)

plt.figure()
plt.grid(linewidth=0.4, color='k')
#plt.plot(t[:,0], t[:,1],'ro', label='uncorrelated in')
plt.plot(ooo[:,0], ooo[:,1], 'bo', label='uncorrelated out')
plt.legend(loc='upper right')
plt.title('get uncorrelated, N input')
plt.xlim(-10,15)
plt.ylim(-10,15)
plt.show()


#------------------------------------------------------#
# testing transformations: direct

u = obj.C2U(tt)
#print u
plt.figure()
plt.grid(linewidth=0.4, color='k')
plt.plot(tt[:,0], tt[:,1], 'bo', label='correlated in')
plt.plot(u[:,0], u[:,1],'ro', label='uncorrelated out')
plt.legend(loc='upper right')
plt.title('direct Nataf')
plt.xlim(-10,15)
plt.ylim(-10,15)
plt.show()

#------------------------------------------------------#
# testing transformations: inverse

c = obj.U2C(u)
#print u
plt.figure()
plt.grid(linewidth=0.4, color='k')
plt.plot(u[:,0], u[:,1], 'bo', label='uncorrelated in')
plt.plot(c[:,0], c[:,1],'ro', label='correlated out')
plt.legend(loc='upper right')
plt.title('inverse Nataf: check the inverse mapping')
plt.xlim(-10,15)
plt.ylim(-10,15)
plt.show()

#------------------------------------------------------#
# testing the mean and the variance of output marginals

print '-----------------------------------------------'
for i in range(len(D)):
    print 'mean of ',i,'th output:', np.mean(u[:,i])
for i in range(len(D)):
    print 'variance of ',i,'th output:', np.var(u[:,i])
print '-----------------------------------------------'

