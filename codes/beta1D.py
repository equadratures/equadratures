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

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk



"""
# Simple analytical function
def fun(x, derivative_flag, error_flag):
    return x[:]**2 + 3*x[:] - 1

def main():

    order = 5 # more for visualization
    derivative_flag = 0 # derivative flag on=1; off=0
    error_flag = 0
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 2, 2 # Jacobi polynomial values for Legendre

    # Uncertainty parameters
    uq_parameter1 = PolynomialParam("Jacobi", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, order) # Setup uq_parameter
    uq_parameters = [uq_parameter1]

    # Compute elements of an index set:self, index_set_type, orders, level=None, growth_rule=None):
    #indexset_configure = IndexSet("total order", [5])
    #indices = IndexSet.getIndexSet(indexset_configure)

    # Create a PolyParent object!
    #uq_structure = PolyParent(uq_parameters, indices)
    #pts, wts = PolyParent.getTensorQuadrature(uq_structure)
    #X = PolyParent.getCoefficients(uq_structure, fun)
    #print X


    A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)
    b = fun(gaussPoints, derivative_flag, error_flag)

    print A
    # Normalize these!
    Aweighted , NormFactor = matrix.rowNormalize(A)
    bweighted = np.dot(NormFactor, b)
    #Cweighted , NormFactorGrad = matrix.rowNormalize(C)
    #dweighted = np.dot(NormFactorGrad, d)

    # Full least squares!
    x_true = matrix.solveLeastSquares(Aweighted, bweighted)

    print x_true
main()
