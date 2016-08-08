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

    def getAmatrix(self):
        return getA(self)

    def getAsubsampled(self, maximum_number_of_evals):
        return getSquareA(self, maximum_number_of_evals)

    def getAwithDerivatives(self):
        return 0

    def normalizeA(self):
        return 0


" The A matrix"
def getA(self):

    stackOfParameters = self.uq_parameters
    polynomial_basis = self.index_set
    dimensions = len(stackOfParameters)
    indices = IndexSet.getIndexSet(polynomial_basis)
    no_of_indices = len(indices)

    # Crate a new PolynomialParam object to get tensor grid points & weights
    polyObject =  PolyParent(stackOfParameters, "tensor grid")
    quadrature_pts, quadrature_wts = PolyParent.getPointsAndWeights(polyObject)

    # Allocate memory for "unscaled points!"
    unscaled_quadrature_pts = np.zeros((len(quadrature_pts), dimensions))

    for i in range(0, dimensions):
        for j in range(0, len(quadrature_pts)):
                if (stackOfParameters[i].param_type == "Uniform"):
                    unscaled_quadrature_pts[j,i] = ((quadrature_pts[j,i] - stackOfParameters[i].lower_bound)/(stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound))*2.0 - 1.0

                elif (stackOfParameters[i].param_type == "Beta" ):
                    unscaled_quadrature_pts[j,i] = (quadrature_pts[j,i] - stackOfParameters[i].lower_bound)/(stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound)

    # Ensure that the quadrature weights sum up to 1.0
    quadrature_wts = quadrature_wts/np.sum(quadrature_wts)
    P = np.mat(PolyParent.getMultivariatePolynomial(polyObject, unscaled_quadrature_pts, indices))
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

def getSquareA(self, maximum_number_of_evals):

    # Get A
    A, quadrature_pts, quadrature_wts = getA(self)
    dimension = len(self.uq_parameters)
    m , n = A.shape

    # Now compute the rank revealing QR decomposition of A!
    P = mat.QRColumnPivoting(A.T)
    selected_quadrature_points = P[0:maximum_number_of_evals]
    Asquare =  mat.getRows(np.mat(A), selected_quadrature_points)
    esq_pts = mat.getRows(np.mat(quadrature_pts), selected_quadrature_points)
    esq_wts = quadrature_wts[selected_quadrature_points]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    return Asquare, esq_pts, W

def error_function(string_value):
    print string_value
    sys.exit()
