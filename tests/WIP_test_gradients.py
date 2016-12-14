#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.computestats import Statistics
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction, evalgradients
import numpy as np

def fun(x):
    return np.cos(x[0]) + np.sin(2*x[1])

def fungrad(x):
    return [-np.sin(x[0]),2*np.cos(2*x[1])] 

# Normalize the rows of A by its 2-norm  
def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization

# Parameters!
pt = 3
x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=pt, derivative_flag=1)
x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=pt, derivative_flag=1)
parameters = [x1, x2]
dims = 2.


# Effective subsampling object!
#esq = EffectiveSubsampling(parameters)



# Get matrices!
#A, p, w = esq.getAmatrix()
#m, n = A.shape
#print m, n
#C = esq.getCmatrix()
#p, q = C.shape

#print p, q
#print C

# Now use the formula to determine the least number of subsamples required!
# Polynomial basis!
hyperbolic_cross = IndexSet("Hyperbolic basis", orders=[pt-1,pt-1], q=1.0)
esq = EffectiveSubsampling(parameters, hyperbolic_cross)
basis_terms_required = 10
s = np.int( (basis_terms_required + dims)/(dims + 1.) )

#maximum_number_of_evals = hyperbolic_cross.getCardinality()

print '################## APPLYING QR w/ Hyperbolic cross ####################'
# Now, lets see what happens if we subsample s number of rows from A using QR!
Asub, esq_pts, W, points = esq.getAsubsampled(s)
f = evalfunction(esq_pts, fun)
df = evalgradients(esq_pts, fungrad)
print Asub
Csub = esq.getCsubsampled(s)

BigA = np.vstack([A, BigC])
    Bigb = np.vstack([b, d])
    print BigA
    print '~~~~~~~~~~~'
    print Bigb
    x = solveLSQ(np.vstack([A, BigC]), Bigb)
#m, n = Asub.shape
#print m, n
#print Csub
#p, q = Csub.shape
#print m, n
#print p, q
#print Asub
#print '\n'
#print Csub

# 1. Stacked least squares!
