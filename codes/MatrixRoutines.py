#!/usr/bin/python
import numpy as np
from scipy.linalg import qr


"""
    Matrix Routines Class

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

    Write something meaningfull here!

"""
# Compute the pivot columns
def QRColumnPivoting(A):
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    return P
