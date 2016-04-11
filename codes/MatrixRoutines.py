#!/usr/bin/python
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
    return A_normalized, normalization_factor
