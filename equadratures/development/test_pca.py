import numpy as np
from equadratures import *
import matplotlib.pyplot as plt

""" testing pca method
    D = list of marginals
"""

D = list()
#D.append(Parameter(order=3, distribution='uniform', lower = 0.01, upper =0.99))
#D.append(Parameter(order=3, distribution='uniform', lower = 0.01, upper =0.99))
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A=2.0, shape_parameter_B=1.))
D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A=2.0, shape_parameter_B=3.9))
#D.append(Parameter(distribution='rayleigh', shape_parameter_A = 1.5, order=5))
#D.append(Parameter(distribution='rayleigh', shape_parameter_A = 12., order=5))
#D.append(Parameter(distribution='beta', shape_parameter_A = 1., shape_parameter_B = 1., lower = 0., upper = 1., order = 5))
#D.append(Parameter(distribution='beta', shape_parameter_A = 2., shape_parameter_B = 2., lower = 0.5, upper = 0.99, order=5))
#D.append(Parameter(distribution='chebyshev', upper = 1., lower=0., order=5))
#D.append(Parameter(distribution='chebyshev', upper = 0.89, lower=0.3, order=5))
#D.append(Parameter(distribution='chisquared', order=5, shape_parameter_A = 14))
#D.append(Parameter(distribution='chisquared',order=5, shape_parameter_A = 2))
#D.append(Parameter(distribution='exponential', order=5, shape_parameter_A = 0.7))
#D.append(Parameter(distribution='exponential', order=5, shape_parameter_A = 0.4))

#D.append(Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = 1., shape_parameter_B = 1., lower = 0.5, upper = 1.5))
#D.append(Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = 1., shape_parameter_B = 1., lower = 0.5, upper = 1.5))


R = np.zeros((len(D),len(D)))
for i in range(len(D)):
    for j in range(len(D)):
        if i==j:
            R[i,j] = 1.0
        else:
            R[i,j] = 0.7

print R
obj = Pca(D,R)

uu = obj.getUncorrelatedSamples(N=1500)
w, u = obj.getCorrelatedSamples(N=1500)

uncorr = obj.C2U(w)

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(w[:,0],w[:,1],'bo',label='first correlation')
plt.plot(u[:,0],u[:,1],'ro',label='original uncorrelated')
plt.legend(loc='upper left')
plt.axis('equal')
plt.show()

print '----------------------------------------------------'
print 'original VS correlation'
print 'mean of original #1:', np.mean(u[:,0]), '#2:', np.mean(u[:,1]) 
print 'mean of correlated #1:', np.mean(w[:,0]), '#2:', np.mean(w[:,1])
print 'variance of original #1:', np.var(u[:,0]), '#2:', np.var(u[:,1])
print 'variance of correlated #1:', np.var(w[:,1]), '#2:', np.var(w[:,1])
print '----------------------------------------------------'

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(w[:,0],w[:,1],'ro',label='correlated')
plt.plot(uncorr[:,0],uncorr[:,1],'bo',label='uncorr: transf.')
plt.legend(loc='upper left')
plt.axis('equal')
plt.show()

print '----------------------------------------------------'
print 'correlated VS direct transf.'
print 'mean of uncorrelated #1:', np.mean(uncorr[:,0]), '#2:', np.mean(uncorr[:,1]) 
print 'mean of correlated #1:', np.mean(w[:,0]), '#2:', np.mean(w[:,1])
print 'variance of uncorrelated #1:', np.var(uncorr[:,0]), '#2:', np.var(uncorr[:,1])
print 'variance of correlated #1:', np.var(w[:,1]), '#2:', np.var(w[:,1])
print '----------------------------------------------------'

corr = obj.U2C(uncorr)

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(corr[:,0], corr[:,1],'ro',label='correlated output')
plt.plot(uncorr[:,0],uncorr[:,1],'bo',label='uncorr input')
plt.legend(loc='upper left')
plt.axis('equal')
plt.show()

print '----------------------------------------------------'
print 'correlatd VS inverse'
print 'mean of uncorrelated #1:', np.mean(uncorr[:,0]), '#2:', np.mean(uncorr[:,1]) 
print 'mean of correlated #1:', np.mean(corr[:,0]), '#2:', np.mean(corr[:,1])
print 'variance of uncorrelated #1:', np.var(uncorr[:,0]), '#2:', np.var(uncorr[:,1])
print 'variance of correlated #1:', np.var(corr[:,1]), '#2:', np.var(corr[:,1])
print '----------------------------------------------------'

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(w[:,0],w[:,1],'ro',label='original correlated')
plt.plot(corr[:,0],corr[:,1],'bo',label='inverse transf.')
plt.legend(loc='upper left')
plt.axis('equal')
plt.show()

print '----------------------------------------------------'
print 'inverse VS original corr.'
print 'mean of correlated original#1:', np.mean(corr[:,0]), '#2:', np.mean(corr[:,1]) 
print 'mean of correlated inverse#1:', np.mean(w[:,0]), '#2:', np.mean(w[:,1])
print 'variance of correlated original#1:', np.var(corr[:,0]), '#2:', np.var(corr[:,1])
print 'variance of correlated inverse#1:', np.var(w[:,1]), '#2:', np.var(w[:,1])
print '----------------------------------------------------'

