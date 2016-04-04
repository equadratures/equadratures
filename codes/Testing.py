#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Testing Class

"""

def main():
    derivative_flag = 1
    order = 5


    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag, order) # Legendre
    A, C = PolynomialParam.getAmatrix(uq_parameter1)
    print(A)
    print(C)



main()
