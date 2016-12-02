#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.computestats import Statistics
import effective_quadratures.qr as qr
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction, evalgradients
import numpy as np

def fun(x):
    return np.exp(x[0])

def fungrad(x):
    return [np.exp(x[0]) ] 

# Normalize the rows of A by its 2-norm  
def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization

def nogradients_univariate():
    
    # Parameters!
    pt = 6
    x1 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt)
    parameters = [x1]

    # Effective subsampling object!
    esq = EffectiveSubsampling(parameters)
    
    # Solve the least squares problem
    A, p, w = esq.getAmatrix()
    W = np.mat(np.diag(np.sqrt(w) ) )
    b = W.T  * evalfunction(p, fun)
    x = qr.solveLSQ( A, b )

def gradients_univariate():
    # Parameters!
    pt = 6
    x1 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt, derivative_flag=1)
    parameters = [x1]

    # Effective subsampling object!
    esq = EffectiveSubsampling(parameters)
    
    # Solve the least squares problem
    A, p, w = esq.getAmatrix()
    C = esq.getCmatrix()
    W = np.mat(np.diag(np.sqrt(w) ) )
    b = W.T  * evalfunction(p, fun)
    d = evalgradients(p, fungrad)
    x = qr.solveLSQ( A, b )

#print p, q
#print C

# Now use the formula to determine the least number of subsamples required!
# Polynomial basis!
#hyperbolic_cross = IndexSet("Hyperbolic basis", orders=[pt-1,pt-1], q=1.0)
#esq = EffectiveSubsampling(parameters, hyperbolic_cross)
#basis_terms_required = hyperbolic_cross.getCardinality() 
#s = np.int( (basis_terms_required + dims)/(dims + 1.) )

#maximum_number_of_evals = hyperbolic_cross.getCardinality()

#print '################## APPLYING QR w/ Hyperbolic cross ####################'
# Now, lets see what happens if we subsample s number of rows from A using QR!
#Asub, esq_pts, W, points = esq.getAsubsampled(s)
#f = W.T * evalfunction(esq_pts, fun)
#df = evalgradients(esq_pts, fungrad, 'vector')
#print f, df
#Csub = esq.getCsubsampled(s)

#x = qr.solveLSQ(np.vstack([Asub, Csub]), np.vstack([f, df])  )
#print x
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
