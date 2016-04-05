#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Optimal Quadrature Subsampling
    1D Example

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    (1) Input function / dimensionality / derivative flag / Maximum number of function evaluations/ Randomized option / Hyperbolic or total order sets?
    (2) Compute the "A" and "C" matrices. If the number of rows & columns exceed a certain threshold automatically use the iterative QR approach
    (3) Depending on whether the random option is selected compute the optimal subsamples -- do this throu

"""

def main():

    #--------------------------------------------------------------------------------------
    #
    #  MAIN INPUTS FOR USER:
    #  Notes:
    #        1. With the derivative flag on we recommend using 2X basis terms
    #
    #--------------------------------------------------------------------------------------
    derivative_flag = 1 # derivative flag on=1; off=0
    basis_terms, quadrature_points = 5 , 6 # basis_terms = order
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 0, 0 # Jacobi polynomial values for Legendre
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, basis_terms) # Setup uq_parameter

    # Compute the A and C matrices
    A, C = PolynomialParam.getAmatrix(uq_parameter1, quadrature_points)

    #


main()
