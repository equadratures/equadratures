#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Testing Class

"""

def main():

    uq_parameter1 = PolynomialParam("Jacobi", -1, 1, 0, 0) # Legendre
    order = 5
    g = PolynomialParam.getRecurrenceCoefficients(uq_parameter1, order)
    print(g)

main()
