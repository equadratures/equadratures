#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Optimal Quadrature subsampling

    1) Input function / dimensionality / derivative flag / Maximum number of function evaluations/ Randomized option / Hyperbolic or total order sets?
    2) Compute the "A" and "C" matrices. If the number of rows & columns exceed a certain threshold automatically use the iterative QR approach
    3) Depending on whether the random option is selected compute the optimal subsamples -- do this throu

"""

def main():

    """
    derivative_flag = 1
    order = 5
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, order) # Legendre
    A, C = PolynomialParam.getAmatrix(uq_parameter1)
    print(A)
    print(C)
    """
    v = [2, 3, 3, 6, 5]
    dims = (2,5)
    g = np.unravel_index(2, dims)
    print(g)

    

main()
