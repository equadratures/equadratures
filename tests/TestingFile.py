#!/usr/bin/python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

"""

    Testing File for Codes

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
# Simple analytical function
def fun(x):
    return x[:]

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION

    NOTES:
        1. min_value and max_value are not used when param_type = "Normal" or
        "Gaussian".
        2. parameter_A and parameter_B are the shape parameters when param_type
        is "Jacobi"; alpha and beta respectively. For a normal distribution
        these become the mean and the standard deviation.
        3. The normal distribution is for a mean of 0 and variance of 0.5 by
        default.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 8
    derivative_flag = 0 # derivative flag on=1; off=0
    error_flag = 0
    min_value, max_value = 0, 1
    parameter_A, parameter_B = 3,2
    uq_parameter1 = PolynomialParam("Jacobi", min_value, max_value, parameter_A, parameter_B, derivative_flag, order) # Setup uq_parameter
    uq_parameters = [uq_parameter1]

    print '************************************'
    print 'min, max, alpha, beta, no. of points, function'
    print min_value, max_value, parameter_A, parameter_B, order, 'y = x'
    # Compute elements of an index set:self, index_set_type, orders, level=None, growth_rule=None):
    indexset_configure = IndexSet("total order", [order])
    indices = IndexSet.getIndexSet(indexset_configure)

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, indices)
    pts, wts = PolyParent.getTensorQuadrature(uq_structure)
    print 'Points'
    print pts
    print 'Weights'
    print wts


#    For coefficients!
    X , T = PolyParent.getCoefficients(uq_structure, fun)
    print '\n Pseudospectral coefficients'
    print X

    print '\n Variance'
    print np.sum(X[0,1:]**2)

    print 'Polynomial evaled at quadrature points'

    #A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)
    #b = fun(gaussPoints, derivative_flag, error_flag)

    #print A
    # Normalize these!
    #Aweighted , NormFactor = matrix.rowNormalize(A)
    #bweighted = np.dot(NormFactor, b)
    #Cweighted , NormFactorGrad = matrix.rowNormalize(C)
    #dweighted = np.dot(NormFactorGrad, d)

    # Full least squares!
    #x_true = matrix.solveLeastSquares(Aweighted, bweighted)

    #print x_true
main()
