#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.computestats import Statistics
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction, evalgradients
import numpy as np
import matplotlib.pyplot as plt
import math
import os
np.set_printoptions(precision=3)

def fun(x):
    return 2*x[0]**2 - 5*x[0]*x[1] + 3*x[1]**3

def fungrad(x):
    return [ 4*x[0] - 5*x[1] , -5*x[0] + 9*x[1]**2 ] 

#def fungradnoised(x):
#    return [np.exp(x[0] + x[1]) - 0.05*np.random.randn(1), np.exp(x[0] + x[1])- 0.05*np.random.randn(1)] 
os.system('clear')
value = 12
x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=value, derivative_flag=1)
x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=value, derivative_flag=1)
parameters = [x1, x2]
hyperbolic_cross = IndexSet("Hyperbolic basis", orders=[value-1,value-1], q=1.0)

###################
# Without gradients!
###################
esq = EffectiveSubsampling(parameters, hyperbolic_cross)
esq.set_no_of_evals(esq.no_of_basis_terms) # in effect making it a square system!
x = esq.computeCoefficients(fun)
print 'Number of basis terms desired in expansion: '+'\t'+str(esq.no_of_basis_terms)

#######################
# Now with gradients
#######################
# 1. Determine least number of subsamples required!
minimum_subsamples = esq.least_no_of_subsamples_reqd()
print 'Number of evaluations used (not counting gradients): '+'\t'+str(minimum_subsamples)

# 2. Set the number of evals to this number!
esq.set_no_of_evals(minimum_subsamples)

# 3. Compute the gradients at the subsamples!
print 'The subsampled quadrature points are:'
print esq.subsampled_quadrature_points

# Store function & gradient values!
fun_values = evalfunction(esq.subsampled_quadrature_points, fun)
grad_values = evalgradients(esq.subsampled_quadrature_points, fungrad, 'matrix')

print '----- Double checking ranks ------'
print esq.A_subsampled.shape
print esq.C_subsampled.shape
print np.linalg.matrix_rank(np.mat( np.vstack([esq.A_subsampled, esq.C_subsampled]), dtype='float64') )
print '----------------------------'

# 5. Compute coefficients using the two techniques!
x1 =  esq.computeCoefficients(fun_values, grad_values, 'weighted')
x1b =  esq.computeCoefficients(fun_values, grad_values, 'equality')
#x1c =  esq.computeCoefficients(fun_values, grad_values, 'equality2')

weighted_error = np.linalg.norm(x-x1, 2)
equality_error = np.linalg.norm(x-x1b, 2)
#equality_error2 = np.linalg.norm(x-x1c, 2)

# 6. Compute 2-norm errors!
print 'Errors:'
print weighted_error, equality_error #, equality_error2
print 'Condition numbers:'
print np.linalg.cond(np.mat( np.vstack([esq.A_subsampled, esq.C_subsampled]), dtype='float64') )
print '\n'
# 7. Prune down the columns
print 'Prune down the number of columns in the matrix'
esq.prune(8)
x1 =  esq.computeCoefficients(fun_values, grad_values, 'weighted')
x1b =  esq.computeCoefficients(fun_values, grad_values, 'equality')
#x1c =  esq.computeCoefficients(fun_values, grad_values, 'equality2')

weighted_error = np.linalg.norm(x[0:esq.no_of_basis_terms]-x1, 2)
equality_error = np.linalg.norm(x[0:esq.no_of_basis_terms]-x1b, 2)
#equality_error2 = np.linalg.norm(x-x1c, 2)

# 6. Compute 2-norm errors!
print 'Errors:'
print weighted_error, equality_error #, equality_error2
print 'Condition numbers:'
print np.linalg.cond(np.mat( np.vstack([esq.A_subsampled, esq.C_subsampled]), dtype='float64') )