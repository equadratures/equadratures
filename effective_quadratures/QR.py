#!/usr/bin/env python
import numpy as np
import scipy.linalg as sc
import Utils as util
"""

    QR factorization routines
    (See MATLAB set of codes in QR-awesomeness repo)

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
        util.error_function('Norm: Either using 1 or 2 as the second argument for the norm')


# Householder reflection
def house(vec):
    m = len(vec)
    sigma = np.dot(vec[1:m].T , vec[1:m] )
    v = np.insert(vec, 0, 1)
    if sigma == 0 and vec[0] >= 0:
        beta = 0
    elif sigma == 0 and vec[0] < 0 :
        beta = -2
    else:
        mu = np.sqrt(vec[0]**2 + sigma)
        if vec[0] <= 0:
            v[0] = vec[0] - mu
        else:
            v[0] = -sigma / (vec[0] + mu)
    beta = 2 * v[0]**2 / (sigma + v[0]**2)
    v = v/v[0]
    v = np.mat(v)
    v = v.T
    return v, beta

"""
    QR Householder with Pivoting!
"""
def qrColumnPivoting_house(A):

    # Determine the size of A
    m = len(A[:,0])
    n = len(A[0,:])
    pivots = np.linspace(0, n-1, n)
    column_norms = np.zeros((n))
    u = np.min([m,n])

    # Set Q and P
    Q = np.mat(np.identity(m))
    R = np.mat(np.zeros((m, n)))

    # Compute the column norms
    for j in range(0, n):
        column_norms[j] = norms(A[:,j], 2)**2

    # Reduction steps
    for k in range(0, u-1):
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
        H = np.mat( np.identity(m-k+1) - beta * np.dot(v, v.T) )
        A[k:m, k:n] = np.dot(H , A[k:m, k:n])
        ## Loop for A[k:m, k:n] = H * A[k:m, k:n]
        #for ii in range(0, m):
        #    for jj in range(0, n):
        #        A[ii, jj] = H[ii, ]

        #if k < m:
        #    A[k+1:m, k] = v[2:m - k + 1]
        print v
        interior = beta * np.mat(np.dot(v, v.T))
        print interior
        print '~~~~~~~~~~~~~~~~~~~~~'
        Q[:,k:m] = Q[:,k:m] - np.dot(Q[:,k:m] , interior)

        # update the remaining column norms
        if k != n:
            for j in range(k + 1, n):
                column_norms[j] = norms(A[0:m, j], 2)**2

    return Q, R, pivots

"""
    Modified Gram Schmidt QR column pivoting!
"""
def qrColumnPivoting_mgs(A):

    # Determine the size of
    m , n = A.shape
    u = np.min([m, n])
    h = np.max([m, n])

    # Set Q and P
    Q = np.mat( np.zeros((m,m)) )
    R = np.mat( np.zeros((m,n)) )

    # Initialize!
    column_norms = np.zeros((n))
    pivots = np.linspace(0,n-1,n)

    # Compute the column norms
    for j in range(0,n):
        column_norms[j] = norms(np.array(A[:,j]), 2)**2


    # Now loop!
    for k in range(0, u):

        #----------------------------------------------
        # Step 0: Column norm sorting
        #----------------------------------------------
        # Find the "j*" column index with the highest
        # column norm
        j_star = np.argmax(column_norms[k:n])
        j_star = j_star + k



        # If j_star = k, skip to step 1, else swap columns!
        if k != j_star:

            # Swap columns in A:
            temp = np.array(A[:,k])
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
        if k != 0:
            for i in range(0,k):
                alpha = np.dot(Q[:,i].T , A[:,k] )
                print R[i,k]
                R[i,k] = R[i,k] + alpha
                print R[i,k]
                A[:,k] = np.array(A[:,k] - alpha[0,0] * Q[:,i])

        #----------------------------------------------
        # Step 2: Normalization
        #----------------------------------------------
        R[k,k] = norms(np.array(A[:,k]), 2)
        Q[:,k] = np.array(A[:,k] * 1.0/R[k,k])

        #----------------------------------------------
        # Step 3: Orthogonalization
        #----------------------------------------------
        if k != n:
            for j in range(k,n):
                R[k,j] = np.dot(Q[:,k].T , A[:,j] )
                A[:,j] = np.array(A[:,j] - np.mat(R[k,j] * Q[:,k]))

                # Now re-compute column norms
                column_norms[j] = norms(np.array(A[:,j]), 2)**2

        # DEBUG: let's print R at each iteration
        print R
        print '***************************************'
    # Ensure that the pivots are integers
    for k in range(0, len(pivots)):
        pivots[k] = int(pivots[k])

    return Q, R, pivots
