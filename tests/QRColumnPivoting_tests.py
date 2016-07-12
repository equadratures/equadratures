#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
"""

    Testing custom QR factorizations with scipy's inbuilt QRP

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
def main():

    A = np.random.rand(10,8)
    print A
    Q, R, P = matrix.qrColumnPivoting_mgs(A)
    print Q, R, P

    Q, R, P = matrix.QRColumnPivoting(A)
    print Q, R, P


main()
