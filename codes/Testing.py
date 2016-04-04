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
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0, derivative_flag) # Legendre
    order = 5
    g = PolynomialParam.getRecurrenceCoefficients(uq_parameter1, order)
    v = PolynomialParam.getOrthoPoly()

main()
