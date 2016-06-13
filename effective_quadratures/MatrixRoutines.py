#!/usr/bin/env python
import numpy as np
import scipy.linalg as sc


"""
    Matrix Routines Class

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    Write something meaningfull here!

"""
# Compute the pivot columns
def QRColumnPivoting(A):
    Q, R, P = sc.qr(A.T,  pivoting=True)
    return P

def solveLeastSquares(A, b):
    rows, cols = A.shape
    rows_b = len(b)
    x = sc.lstsq(A, b)
    return x[0]

def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization

"""
    MODIFIED GRAM SCHMIDT QR COLUMN PIVOTING
    INPUTS:
        fun_cols_A(j): A function that returns the jth column of A
        m : number of rows in A
        n : number of columns in A

    OUTPUTS:
        P : pivots
"""
def qrColumnPivoting_mgs(fun_cols_A, m, n):

    # Compute the column norms of A:
    column_norms = np.zeros((n))
    for j in range(0,n):
        column_norms[j] = np.sum(fun_cols_A(j)**2)

    # Now loop!
    for k in range(0, n):

        #----------------------------------------------
        # Step 0
        #----------------------------------------------
        # Find the "j*" column index with the highest
        # column norm
        value, j_star = np.max(column_norms[k:n])
        j_star = j_star + k

        if k != j_star:

            # Swap columns in A:
            temp = fun_cols_A(k)
            A_k = fun_cols_A(j_star)
            A_j_star = temp

        # Swap columns in R accordingly
