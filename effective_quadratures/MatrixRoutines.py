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
        A: matrix A

    OUTPUTS:
        Q: orthogonal matrix
        R: upper triangular matrix
        P: pivots

    References:
    1. A. Dax
    2. Golub, G., VanLoan, C., "Matrix Computations"
"""
def qrColumnPivoting_mgs(A):

    # Determine the size of A
    m = len(A[:,0])
    n = len(A[0,:])
    h = np.max([m, n])

    # Set Q and P
    Q = np.mat( np.zeros((m,n)) )
    R = np.mat( np.zeros((n,n)) )

    # Initialize!
    column_norms = np.zeros((n))
    pivots = np.linspace(0,h-1,h)
    print pivots

    # Compute the column norms
    for j in range(0,n):
        column_norms[j] = np.sum(A[:,j]**2)

    # Now loop!
    for k in range(0, n):

        #----------------------------------------------
        # Step 0: Column norm sorting
        #----------------------------------------------
        # Find the "j*" column index with the highest
        # column norm
        j_star = np.argmax(column_norms[k:n])
        j_star = j_star + k

        if k != j_star:

            # Swap columns in A:
            temp = A[:,j]
            A[:,k] = A[:,j_star]
            A[:,j_star] = temp

            # Swap columns in R accordingly
            for i in range(0, k-1):
                temp = R[i,k]
                R[i,k] = R[i, j_star]
                R[i, j_star] = temp

            # Swap pivots
            temp = pivots[k]
            pivots[k] = pivots[j_star]
            pivots[j_star] = temp


        #-----------------------------------------------
        # Step 1: Reorthogonalization
        #-----------------------------------------------
        if k != 1:
            for i in range(0,k-1):
                alpha = np.dot(Q[:,i].T , A[:,k] )
                R[i,k] = R[i,k] + alpha
                for u in range(0,m):
                    A[u,k] = A[u,k] - alpha * Q[u,i]

        #----------------------------------------------
        # Step 2: Normalization
        #----------------------------------------------
        R[k,k] = np.linalg.norm(A[:,k], 2)
        for i in range(0, m):
            Q[i,k] = A[i,k] * 1.0/R[k,k]

        #----------------------------------------------
        # Step 3: Orthogonalization
        #----------------------------------------------
        if k != n:
            for j in range(k+1,n):
                R[k,j] = np.dot(Q[:,k].T , A[:,j] )
                for v in range(0,m):
                    A[v,j] = A[v,j] - R[k,j] * Q[v,k]

                # Now re-compute column norms
                column_norms[j] = np.linalg.norm(A[:,j]**2, 2)

    return Q, R, pivots
