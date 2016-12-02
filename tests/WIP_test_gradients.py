#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.computestats import Statistics
import effective_quadratures.qr as qr
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction, evalgradients
import numpy as np
" Template for deciding how best to code effectivequads.py"

def getRows(A, row_indices):
    
    # Determine the shape of A
    m , n = A.shape

    # Allocate space for the submatrix
    A2 = np.zeros((len(row_indices), n))

    # Now loop!
    for i in range(0, len(A2)):
        for j in range(0, n):
            A2[i,j] = A[row_indices[i], j]

    return A2

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
    A, p, w = esq.getAmatrix() # Is this always square??
    W = np.mat(np.diag(np.sqrt(w) ) )
    b = W.T  * evalfunction(p, fun)
    x = qr.solveLSQ(A,b)

def gradients_univariate():
    
    # Parameters!
    pt = 6
    x1 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt, derivative_flag=1)
    x2 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt, derivative_flag=1)
    parameters = [x1, x2]
    dims = 1

    # Basis selection!
    hyperbolic_cross = IndexSet("Hyperbolic basis", orders=[pt-1,pt-1], q=1.0)
    esq = EffectiveSubsampling(parameters, hyperbolic_cross)
    A , p, w = esq.getAmatrix()

    basis_terms_required = hyperbolic_cross.getCardinality() 
    s = np.int( (basis_terms_required + dims)/(dims + 1.) ) 
    
    # Now do QR column pivoting on A
    P = qr.mgs_pivoting(A.T)
    selected_quadrature_points = P[0:s]
    Afat =  getRows(np.mat(A), selected_quadrature_points)

    # What is the size of Asub?
    m, n = Afat.shape
    esq_pts = getRows(np.mat(quadrature_pts), selected_quadrature_points)
    esq_wts = quadrature_wts[selected_quadrature_points]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    print m, n

    f = W.T * evalfunction(esq_pts, fun)
    #df = evalgradients(esq_pts, fungrad, 'vector')
    #Csub = esq.getCsubsampled(s)
    #x = qr.solveLSQ(np.vstack([Asub, Csub]), np.vstack([f, df])  )
    #print x

gradients_univariate()