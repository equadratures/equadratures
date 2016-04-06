#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import MatrixRoutines as matrix
import matplotlib.pyplot as plt
import numpy as np
"""
    Optimal Quadrature Subsampling
    1D Example

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

"""

# Simple analytical function
def fun(x):
    return np.exp(x[:])


def main():

    #--------------------------------------------------------------------------------------
    #
    #  USER'S NOTES:
    #        1. With the derivative flag on we recommend using 2X basis terms
    #        2. Input maximum number of permissible model evaluations
    #        3. Input number of points on the "full grid" (3x5 times number in line above)
    #
    #--------------------------------------------------------------------------------------
    derivative_flag = 0 # derivative flag on=1; off=0
    evaluations_user_can_afford = 8 # basis_terms = order

    # Determine the number of basis terms
    if derivative_flag == 1:
        basis_terms = 2 * evaluations_user_can_afford
    else:
        basis_terms = evaluations_user_can_afford

    full_grid_points = 8 # full tensor grid
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 0, 0 # Jacobi polynomial values for Legendre
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, full_grid_points) # Setup uq_parameter

    # Compute A and C matrices!
    A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)

    # Pick select columns. This amounts using either a total order or hyperbolic cross
    # basis set in nD
    Atall = A[:, 0 : evaluations_user_can_afford]

    # Now compute the "optimal" subsamples from this grid!
    P = matrix.QRColumnPivoting(Atall)
    P = P[0:evaluations_user_can_afford]

    # Now take the first "evaluations_user_can_afford" rows from P
    Asquare = Atall[P,:]

    # Row normalize the matrix!
    Asquare_norms = np.sqrt(np.sum(Asquare**2, axis=1)/(1.0 * evaluations_user_can_afford))
    Normalization_Constant = np.diag(1.0/Asquare_norms)
    A_LSQ = np.dot(Normalization_Constant, Asquare)
    b_LSQ = np.dot(Normalization_Constant, fun(gaussPoints[P,:] )  )

    # Solve the least squares problem
    x = matrix.solveLeastSquares(A_LSQ, b_LSQ)
    print(x)


main()
