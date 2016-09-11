#!/usr/bin/env python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from QR import mgs_pivoting
from IndexSets import IndexSet
import Utils as utils
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

    def getAsubsampled(self, maximum_number_of_evals, flag=None):
        return getSquareA(self, maximum_number_of_evals, flag)

    def getAwithDerivatives(self):
        return 0

# A matrix formed by a tensor grid of rows and a user-defined set of columns.
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

# The subsampled A matrix based on either randomized selection of rows or a QR column pivoting approach
def getSquareA(self, maximum_number_of_evals, flag=None):

    if flag == "QR" or flag is None:
        option = 1 # default option!
    elif flag == "Random":
        option = 2
    else:
        utils.error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): For the third input choose from either 'QR' or 'Random'")

    A, quadrature_pts, quadrature_wts = getA(self)
    dimension = len(self.uq_parameters)
    m , n = A.shape

    if maximum_number_of_evals < n :
        print 'Dimensions of A prior to subselection:'
        print m, n
        print 'The maximum number of evaluations you requested'
        print maximum_number_of_evals
        utils.error_function("ERROR in EffectiveQuadSubsampling --> getAsubsampled(): The maximum number of evaluations must be greater or equal to the number of basis terms")


    # Now compute the rank revealing QR decomposition of A!
    if option == 1:
        P = mgs_pivoting(A.T)
        #P = mat.QRColumnPivoting(A.T)
        #print P
    else:
        P = np.random.randint(0, len(quadrature_pts) - 1, len(quadrature_pts) - 1 )

    # Now truncate number of rows based on the maximum_number_of_evals
    selected_quadrature_points = P[0:maximum_number_of_evals]
        
    # Form the "square" A matrix.
    Asquare =  mat.getRows(np.mat(A), selected_quadrature_points)
    esq_pts = mat.getRows(np.mat(quadrature_pts), selected_quadrature_points)
    esq_wts = quadrature_wts[selected_quadrature_points]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    return Asquare, esq_pts, W, selected_quadrature_points

# Returns only the gradients!?
# def getAwithC(self)

def error_function(string_value):
    print string_value
    sys.exit()
