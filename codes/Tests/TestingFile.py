#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import MatrixRoutines as matrix
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
    return x[:]**2 + 3*x[:] - 1

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION

    NOTES:
        1. min_value and max_value are not used when param_type = "Normal" or
        "Gaussian".
        2. parameter_A and parameter_B are the shape parameters when param_type
        is "Jacobi", alpha and beta respectively. For a normal distribution
        these become the mean and the standard deviation.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 8
    derivative_flag = 0 # derivative flag on=1; off=0
    error_flag = 0
    min_value, max_value = -1, 1
    parameter_A, parameter_B = 0, 0
    uq_parameter1 = PolynomialParam("Normal", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, order) # Setup uq_parameter
    uq_parameters = [uq_parameter1]

    # Compute elements of an index set:self, index_set_type, orders, level=None, growth_rule=None):
    indexset_configure = IndexSet("total order", [order])
    indices = IndexSet.getIndexSet(indexset_configure)

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, indices)
    pts, wts = PolyParent.getTensorQuadrature(uq_structure)
    print pts

#    For coefficients!
#    X = PolyParent.getCoefficients(uq_structure, fun)
#    print X


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
