#!/usr/bin/env python
import numpy as np
import scipy.linalg as sc
import Utils as util
"""

    QR factorization routines
    Based on QR-awesomeness

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""

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

"""
    QR Householder with Pivoting!
"""
# Householder reflection
def house(x):
    m = len(vec)
    sigma = np.dot(vec(1:m).T , vec(1:m) )
    v =

    return v, beta

def qrColumnPivoting_house(A):

    # Determine the size of A
    m = len(A[:,0])
    n = len(A[0,:])
    pivots = np.linspace(0, n-1, n)
    column_norms = np.zeros((n))

    # Set Q and P
    Q = np.mat(np.zeros((m, m)))
    R = np.mat(np.zeros((m, n)))

    # Compute the column norms
    for j in range(0, n):
        column_norms[j] = norms(A[:,j], 2)**2

    # Reduction steps
    for k in range(0, u-1)
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

        # Reduction
        v, beta = house(A[k:m, k])
        H = np.identity(m-k+1) - beta * np.dot(v, v.T)
        A[k:m, k:n] = H * A[k:m, k:n]
        if k < m:
            A[k+1:m, k] = v[2:m - k + 1]

        Q[:,k:m] = Q[:,k:m] - Q[:,k:m] * np.dot(v, v.T) * beta

        # update the remaining column norms
        if k != n
            for j in range(k + 1, n)
                column_norms[j] = norms(A[1:m, j], 2)**2

    return Q, R, pivots



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
