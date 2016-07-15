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
    return Q, R, P

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

" Norm computation"
def norms(vector, integer):

    # l1 norm
    if integer == 1:
        return np.sum(np.abs(vector))
    elif integer == 2:
        # l2 norm
        return np.sqrt(np.sum(vector**2))
    else:
        util.error_function('NORM: DO NOT RECOGNIZE SECOND ARGUMENT!')

def rowNormalize(A):
    rows, cols = A.shape
    A_norms = np.sqrt(np.sum(A**2, axis=1)/(1.0 * cols))
    Normalization = np.diag(1.0/A_norms)
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization


"""
    QR Householder with Pivoting!
"""
def qrColumnPivoting_house(A):

    # Determine the size of A
    m = len(A[:,0])
    n = len(A[0,:])

    return 0



"""
    Modified Gram Schmidt QR column pivoting!
"""
def qrColumnPivoting_mgs(A):

    # Determine the size of A
    m = len(A[:,0])
    n = len(A[0,:])
    u = np.min([m, n])
    h = np.max([m, n])

    # Set Q and P
    Q = np.mat( np.zeros((m,n)) )
    R = np.mat( np.zeros((n,n)) )

    # Initialize!
    column_norms = np.zeros((n))
    pivots = np.linspace(0,n-1,n)

    # Compute the column norms
    for j in range(0,n):
        column_norms[j] = norms(A[:,j], 2)**2

    # Now loop!
    for k in range(0, u):

        #----------------------------------------------
        # Step 0: Column norm sorting
        #----------------------------------------------
        # Find the "j*" column index with the highest
        # column norm
        j_star = np.argmax(column_norms[k:n])
        j_star = j_star + k - 1

        # If j_star = k, skip to step 1, else swap columns!
        if k != j_star:

            # Swap columns in A:
            temp = A[:,j]
            A[:,k] = A[:,j_star]
            A[:,j_star] = temp
            del temp

            # Swap columns in R accordingly
            for i in range(0, k-1):
                temp = R[i,k]
                R[i,k] = R[i, j_star]
                R[i, j_star] = temp
                del temp

            # Swap pivots
            temp = pivots[k]
            pivots[k] = pivots[j_star]
            pivots[j_star] = temp
            del temp

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
        R[k,k] = norms(A[:,k], 2)
        for i in range(0, m):
            Q[i,k] = A[i,k]/R[k,k]

        #----------------------------------------------
        # Step 3: Orthogonalization
        #----------------------------------------------
        if k != n:
            for j in range(k+1,n):
                R[k,j] = np.dot(Q[:,k].T , A[:,j] )
                for v in range(0,m):
                    A[v,j] = A[v,j] - R[k,j] * Q[v,k]
                # Now re-compute column norms
                column_norms[j] = norms(A[:,j], 2)**2

    # Ensure that the pivots are integers
    for k in range(0, len(pivots)):
        pivots[k] = int(pivots[k])

    return Q, R, pivots
