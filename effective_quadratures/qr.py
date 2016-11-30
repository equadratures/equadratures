#!/usr/bin/env python
"""Utilities with QR factorization"""
import numpy as np
from utils import error_function
#****************************************************************************
# Functions to code:
#    
# 1. Block & Block Recursive QR
# 2. Constrained & weighted least squares - direct elimination & null space
# 3. Singular value decomposition & bidiagonalization
# 4. Routines for total least squares problem
# 5. Rank 1 updates for QR factorization
# 6. The 'Practical QR' algorithm
# 7. Randomized QR 
#****************************************************************************
def solve_constrainedLSQ(A,b,C,d):
    """
    Solves the direct, constraint least squares problem ||Ax-b||_2 subject to Cx=d using 
    the method of direct elimination

    :param numpy matrix A: an m-by-n matrix
    :param numpy matrix b: an m-by-1 matrix
    :param numpy ndarray C: an k-by-n matrix
    :param numpy ndarray d: an k-by-1 matrix
    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray

    """
    A = np.mat(A)
    dimensions = len(C)
    C0 = C[0] # Which by default has to exist!
    rows, cols = C0.shape
    BigC = np.zeros((rows*dimensions, cols))
    counter = 0
    dims = 0
    while dims < dimensions:
        for i in range(0, rows):
            for j in range(0, cols):
                BigC[counter, j] = C[dims][i,j]
            counter = counter + 1 
        dims = dims + 1

    # BigC stacks all the C matrices 
    BigC = np.mat(BigC)
   
    # Size of matrices!
    m, n = A.shape
    p, q = b.shape
    k, l = BigC.shape
    s, t = d.shape

    print k, s
    print m, n
    
    # Check that the number of elements in b are equivalent to the number of rows in A
    if m != p:
        error_function('ERROR: mismatch in sizes of A and b')
    elif k != s:
        error_function('ERROR: mismatch in sizes of C and d') 
    
    # Method 1: Stacked least squares approach!
    BigA = np.vstack([A, BigC])
    Bigb = np.vstack([b, d])
    print BigA
    print '~~~~~~~~~~~'
    print Bigb
    x = solveLSQ(BigA, Bigb)
    

    #Q , R = qr_Householder(BigC, 1) # Thin QR factorization on C'
    #R = R[0:n, 0:n]
    #u = np.linalg.inv(R.T) * d
    #A_hat = A * Q
    #z, w = A_hat.shape
    
    # Now split A
    #Ahat_1 = A_hat[:, 0:len(u)]
    #Ahat_2 = A_hat[:, len(u) : w]
    #r = b - Ahat_1 * u
    
    # Solve the least squares problem
    #v = solveLSQ(Ahat_2, r)
    #x = Q * np.vstack([u, v]) 
    
    return x

# Solve the least squares problem (done wiht QR factorization!)
# Perhaps also add LU and SVD techniques to solve least squares!
def solveLSQ(A, b):
    """
    Solves the direct least squares problem ||Ax-b||_2 using the method of QR factorization

    :param numpy ndarray A: an m-by-n A matrix
    :param numpy ndarray b: an m-by-1 b matrix
      
    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray

    """
    # Direct methods!
    Q, R = qr_Householder(A, 1)
    x = np.linalg.inv(R) * Q.T * b
    x = np.array(x)
    return x

# Householder reflection
def house(vec):
    """
    Returns a scalar and a vector that may be used to form a Householder matrix

    :param numpy ndarray vec: The input vector that needs to be reflected in the hyperplane
        spanned by the Householder vector, v

    :return: v, the Householder vector
    :rtype: ndarray
    :return: beta, the Householder scalar
    :rtype: double

    **Notes**
        The Householder vector is given by P = I - beta * v * v'

    """
    m = len(vec)
    sigma = np.dot(vec[1:m].T , vec[1:m] )
    v = np.insert(vec[1:m], 0, 1.0)

    if len(vec) == 1:
        v[0,0] = 1.0
        betav = vec[0,0]
    elif sigma == 0 and vec[0,0] >= 0:
        beta = 0.0
    elif sigma == 0 and vec[0,0] < 0 :
        beta = -2.0
    else:
        mu = np.sqrt(vec[0,0]**2.0 + sigma)
        if vec[0,0] <= 0:
            v[0,0] = vec[0,0] - mu
        else:
            v[0,0] = ( -1.0 * sigma) / (mu + vec[0,0]*1.0 )
    beta = (2.0 * v[0,0]**2) / (1.0 * sigma + v[0,0]**2.0)
    v = v/(1.0 * v[0,0])
    v = np.mat(v)
    v = v.T
    return v, beta

# QR factorization via the method of Householder
def qr_Householder(A, thin=None):
    """
    Returns the Householder QR factorization of a matrix A using the method of Householder

    :param numpy matrix A: Matrix input for QR factorization
    :param boolean thin: Set thin to 1 to compute a thin QR factorization. The default is a regular QR factorization.
    :return: Q, the orthogonal matrix
    :rtype: numpy matrix
    :return: R, the upper triangular matrix
    :rtype: numpy matrix

    **Sample declaration**
    :: 
        >> Q, R = qr_Householder(A)
    """
    A = np.mat(A)
    m, n = A.shape
    k = min([m,n])
    Q = np.identity(m)

    for j in range(0, k):
        v, betav = house(A[j:m, j])
        K = np.multiply(betav,  v * (v.T) )
        H = (np.identity(m-j) - K )
        A[j:m, j:n] =  H * A[j:m, j:n]
        Q[:, j:m] = Q[:, j:m] - Q[:, j:m] * K

    R = np.triu(A)

    # For making it thin!
    if thin == 1:
        Q = Q[0:m, 0:n]
        R = R[0:n, 0:n]
    
    Q = np.mat(Q)
    R = np.mat(R)
    return Q, R

# Modified Gram Schmit QR column pivoting
def mgs_pivoting(A):
    """
    Modified Gram Schmidt QR Column pivoting

    :param numpy matrix A: Matrix input for QR factorization
    :return: pivots: Index of pivoted columns
    :rtype: numpy ndarray

    """
    # Determine the size of
    A = np.matrix(A)
    m , n = A.shape
    u = np.min([m, n])
    h = np.max([m, n])

    # Initialize!
    column_norms = np.zeros((n))
    pivots = range(0, n)

    # Compute the column norms
    for j in range(0,n):
        column_norms[j] = np.linalg.norm(A[0:m,j], 2)**2

    # Now loop!
    for k in range(0, u):

        # Compute the highest norm
        j_star = np.argmax(column_norms[k:n])
        r_star = j_star + k 

        # Retrieve the k-th column of A
        a_k = A[0:m,k]
        
        # Swaping routine
        if k != r_star:
            A[0:m, [r_star, k]] = A[0:m, [k, r_star]]
            temp = pivots[r_star]
            pivots[r_star] = pivots[k]
            pivots[k] = temp

        # orthogonalization
        if k != n:
            for j in range(k+1, n):
                a_j = A[0:m,j]
                intermediate_vec = np.multiply(1.0/(1.0 * np.linalg.norm(a_k, 2) ) , a_k)
                a_j = a_j -  np.multiply( (intermediate_vec.T * a_j) , intermediate_vec )
                A[0:m,j] = a_j

                # update remaining column norms
                column_norms[j] = np.linalg.norm( A[0:m,j] , 2 )**2

                # updating using pythogorean rule! --- do not use! ---
                # temp =  (intermediate_vec.T * a_j) 
                # column_norms[j] = column_norms[j]**2  - (temp / np.linalg.norm(a_j, 2) )**2
                
       # re-orthogonalization
        if k != 0:
            for i in range(0, k-1):
                a_i = A[0:m, i]
                intermediate_vec = np.multiply(1.0/(1.0 *  np.linalg.norm(a_i, 2) ), a_i)
                a_k = a_k - np.multiply( (intermediate_vec.T * a_k) , intermediate_vec)
                del intermediate_vec

        # Final update.
        A[0:m,k] = a_k
        del a_k
        
    return pivots