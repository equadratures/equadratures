

#from equadratures.analyticaldistributions import *
#from equadratures.indexset import *
#from equadratures.parameter import *
#from equadratures.plotting import *
#from equadratures.polyint import *
#from equadratures.polylsq import *
#from equadratures.polyreg import *
#from equadratures.qr import *
#from equadratures.stats import *
#from equadratures.utils import *
from equadratures_test import *
import numpy as np

def fun(x):
    return x[0] + x[1]*x[0] + x[2]*x[1]*x[0]

#Input vars

degree = 4
points_used = degree + 1
mu = 1
sigma = 2
variance = sigma**2
x1 = Parameter(param_type="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, points=points_used)
x2 = Parameter(param_type="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, points=points_used)
x3 = Parameter(param_type="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, points=points_used)
parameters = [x1, x2, x3]

#Polynomial chaos

basis = IndexSet('Tensor grid',[degree, degree, degree])
uqProblem = Polyint(parameters, basis)
pts, wts = uqProblem.getPointsAndWeights()

coefficients, indices, pts = uqProblem.getPolynomialCoefficients(fun)
#print(basis.elements)
stats = Statistics(coefficients, basis)
#print stats.mean
print stats.variance
print stats.fosi
print stats.sobol
print sum(stats.sobol.values())
#print sum(stats.sobol)

#c = coefficients>0.1
#c = c.flatten()
#print indices[c]
#print coefficients[c]

#plotting.coeffplot2D(coefficients, basis.elements,r'$i_1$', r'$i_2$')

#MC
#
#large_number = 1000000
#s = sigma * np.random.randn(large_number,2) + mu
#f = np.zeros((large_number,1))
#for i in range(0, large_number):
#    f[i,0] = fun([s[i,0], s[i,1]])
#    
#print 'MONTE CARLO MEAN & VARIANCE:'
#print str(np.mean(f))+'\t'+str(np.var(f))