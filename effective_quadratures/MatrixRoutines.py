#!/usr/bin/env python
import numpy as np
import scipy.linalg as sc
import Utils as util

"""
    Useful matrix routines.
"""
# Compute the pivot columns
def QRColumnPivoting(A):
    Q, R, P = sc.qr(A,  pivoting=True)
    return P

def solveLeastSquares(A, b):
    rows, cols = A.shape
    rows_b = len(b)
    x = sc.lstsq(A, b)
    return x[0]

# Function that returns a submatrix of specific rows
def getRows(A, row_indices):
    m = len(A) # number of rows
    n = len(A[0,:]) # number of columns

    # Allocate space for the submatrix
    A2 = np.zeros((len(row_indices), n))

    # Now loop!
    for i in range(0, len(A2)):
        for j in range(0, n):
            A2[i,j] = A[row_indices[i], j]

    return A2

def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization
