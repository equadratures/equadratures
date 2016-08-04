#!/usr/bin/env python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import numpy as np
import sys
import MatrixRoutines as mat

"""
    Effectively Subsamples Quadratures Class

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    WARNING: Coding in progress!

"""

class EffectiveSubsampling(object):

    def __init__(self, uq_parameters, index_set, derivative_flag):

        self.uq_parameters = uq_parameters
        self.index_set = index_set

    def getAmatrix(self,points=None):
        return getA(self, points=None)

    def getAsubsampled(self, maximum_number_of_evals, points=None):
        return getSquareA(self, maximum_number_of_evals, points)

    def getAwithDerivatives(self):
        return 0

    def normalizeA(self):
        return 0


" The A matrix"
def getA(self, points):

    stackOfParameters = self.uq_parameters
    polynomial_basis = self.index_set
    dimensions = len(stackOfParameters)
    indices = IndexSet.getIndexSet(polynomial_basis)
    no_of_indices = len(indices)

    # Crate a new PolynomialParam object to get tensor grid points & weights
    polyObject =  PolyParent(stackOfParameters, "tensor grid")
    quadrature_pts, quadrature_wts = PolyParent.getPointsAndWeights(polyObject)

    # Check for points
    if points == None:
        quadrature_pts = quadrature_pts
    else:
        quadrature_pts = points

    P = np.mat(PolyParent.getMultivariatePolynomial(polyObject, quadrature_pts, indices))
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

def getSquareA(self, maximum_number_of_evals, points):

    # Get A
    A, quadrature_pts, quadrature_wts = getA(self, points)
    # Determine the size of A
    m , n = A.shape
    #print m, n
    #print m, n
    # Check that A is a tall matrix!
    #if m < n:
    #    error_function('ERROR: For QR column pivoting, we require m > n!')

    # Now compute the rank revealing QR decomposition of A!
    P = mat.QRColumnPivoting(A.T)
    selected_quadrature_points = P[0:maximum_number_of_evals]
    Asquare =  mat.getRows(np.mat(A), selected_quadrature_points)
    esq_pts = mat.getRows(np.mat(quadrature_pts), selected_quadrature_points)
    esq_wts = quadrature_wts[selected_quadrature_points]
    W = np.mat(np.diag(0.5 * np.sqrt(esq_wts)))
    return Asquare, esq_pts, W

def error_function(string_value):
    print string_value
    sys.exit()
