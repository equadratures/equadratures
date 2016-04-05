#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import MatrixRoutines as matrix
import matplotlib.pyplot as plt

"""
    Optimal Quadrature Subsampling
    1D Example

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    Write something meaningfull here!

"""

def main():

    #--------------------------------------------------------------------------------------
    #
    #  USER'S NOTES:
    #        1. With the derivative flag on we recommend using 2X basis terms
    #        2. Input number of points on the "full grid"
    #        3. Input maximum number of permissible model evaluations
    #
    #--------------------------------------------------------------------------------------
    derivative_flag = 1 # derivative flag on=1; off=0
    evaluations_user_can_afford = 5 # basis_terms = order

    # Determine the number of basis terms
    if derivative_flag == 1:
        basis_terms = 2 * evaluations_user_can_afford
    else:
        basis_terms = evaluations_user_can_afford

    full_grid_points = 12
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 0, 0 # Jacobi polynomial values for Legendre
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, full_grid_points) # Setup uq_parameter

    # Compute the A and C matrices
    A, C = PolynomialParam.getAmatrix(uq_parameter1, quadrature_points)

    # Now compute the "optimal" subsamples from this grid!
    optimalSubsamples = matrix.QR

main()
