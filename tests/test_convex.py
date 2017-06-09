#!/usr/bin/env python
from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestConvex(TestCase):
    
    no_of_subsamples = 4
    difference = 1
    x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
    x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
    x3 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
    parameters = [x1, x1, x1, x1, x1]
    basis = no_of_subsamples - difference
    Hyperbolic = IndexSet("Hyperbolic basis", orders=[basis, basis, basis, basis, basis], q=0.3)
    
    
    esq = Polylsq(parameters, Hyperbolic, 'Random')
    esq.set_no_of_evals(esq.least_no_of_subsamples_reqd() )
    A_qr = esq.A_subsampled

    esq2 = Polylsq(parameters, Hyperbolic, 'Convex')
    esq2.set_no_of_evals(esq2.least_no_of_subsamples_reqd() )
    A_cx = esq2.A_subsampled

    print 'Condition number of QR:'+str(np.linalg.cond(A_qr) )
    print 'Condition number of CVX:'+str(np.linalg.cond(A_cx))

    print 'Error in QR:'+str(np.linalg.norm(A_qr.T * A_qr - np.eye(esq2.least_no_of_subsamples_reqd() ) , 'fro'))
    print 'Error in CVX:'+str(np.linalg.norm(A_cx.T * A_cx - np.eye(esq2.least_no_of_subsamples_reqd() ) , 'fro'))
    #np.savetxt('test.out', esq.A, delimiter=',',  fmt='%1.6e')
    print esq.A.shape

if __name__ == '__main__':
    unittest.main()
