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


# We assume here that C is output as a cell!
def getRowsFromCell(G, row_indices):
    small_rows = len(row_indices)
    dimensions = len(G)
    G0 = G[0] # Which by default has to exist!
    C0 = G0.T
    C0 = getRows(C0, row_indices)
    rows, cols = C0.shape
    BigC = np.zeros((dimensions*rows, cols))
    counter = 0
    for i in range(0, dimensions):
        K = G[i].T
        K = getRows(K, row_indices)
        for j in range(0, rows):
            for k in range(0,cols):
                BigC[counter,k] = K[j,k]
            counter = counter + 1 
    BigC = np.mat(BigC)
    return BigC
    

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
    return np.exp(x[0] + x[1])

def fungrad(x):
    return [np.exp(x[0] + x[1]), np.exp(x[0] + x[1]) ] 

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
    pt = 5
    x1 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt, derivative_flag=1)
    x2 = Parameter(param_type="Uniform", lower=-1.0, upper=1.0, points=pt, derivative_flag=1)
    parameters = [x1, x2]
    dims = len(parameters)

    # Basis selection!
    hyperbolic_cross = IndexSet("Total order", orders=[pt-1,pt-1])
    esq = EffectiveSubsampling(parameters, hyperbolic_cross)
    A , p, w = esq.getAmatrix()
    C = esq.getCmatrix()

    basis_terms_required = hyperbolic_cross.getCardinality() 
    minimum_points = np.int( (basis_terms_required + dims)/(dims + 1.) ) + 4
    
    # Now do QR column pivoting on A
    P = qr.mgs_pivoting(A.T)
    selected_nodes = P[0:minimum_points]
    Afat =  getRows(np.mat(A), selected_nodes)
    m , n = Afat.shape
    print m , n    
    esq_pts = getRows(np.mat(p), selected_nodes)
    esq_wts = w[selected_nodes]
    W = np.mat(np.diag(np.sqrt(esq_wts)))

    # Function and gradient evaluation!
    f = W.T * evalfunction(esq_pts, fun)
    df = evalgradients(esq_pts, fungrad, 'vector')

    Ccell = esq.getCmatrix()
    #print Ccell
    Cfat = getRowsFromCell(Ccell, selected_nodes)
    #print '\n'
    #print '************'
    print selected_nodes
    #print Cfat
    #print '\n'
    m, n = Cfat.shape
    print m, n

    # Now solve the least squares problem!
    print np.vstack([Afat, Cfat])
    x = qr.solveLSQ(np.vstack([Afat, Cfat]), np.vstack([f, df])  )
    print x


gradients_univariate()