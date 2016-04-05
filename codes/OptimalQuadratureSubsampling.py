#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Optimal Quadrature Subsampling

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    1) Input function / dimensionality / derivative flag / Maximum number of function evaluations/ Randomized option / Hyperbolic or total order sets?
    2) Compute the "A" and "C" matrices. If the number of rows & columns exceed a certain threshold automatically use the iterative QR approach
    3) Depending on whether the random option is selected compute the optimal subsamples -- do this throu

"""

def main():


    derivative_flag = 1
    order, quadrature_points = 5 , 6
    min_value, max_value = -1, 1
    alpha_parameter, beta_parameter = 0, 0

    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, order) # Legendre
    A, C = PolynomialParam.getAmatrix(uq_parameter1, quadrature_points)
    print(A)
    print ('****')
    print(C)



main()
