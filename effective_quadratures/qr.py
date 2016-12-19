#!/usr/bin/env python
"""Utilities with QR factorization"""
import numpy as np
from utils import error_function
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
#****************************************************************************
# Functions to code:
#    
# 1. Block & Block Recursive QR (see GVL and Demmel)
# 2. Constrained & weighted least squares - direct elimination & null space
# 3. Singular value decomposition & bidiagonalization
# 4. Routines for total least squares problem
# 5. Rank 1 updates for QR factorization
# 6. The 'Practical QR' algorithm
# 7. Randomized QR 


# Bidiagonalization, symmetric QR, CS decomposition, SVD, gSVD!
#****************************************************************************
def gsvd(A,B):
    """
    Computes a generalized singular value decomposition
    """
    m, n = A.shape


    # Step 1. Compute a QR factorization of [A;B]
    Q, R = qr_MGS(np.vstack([A, B]))

    # Step 2. CS decomposition of 

    return 0

def svd(A, eps):
    """
    Computes the singular value decomposition of a matrix based on Golub-Kahan
    """

    # Step 1. Compute a bidiagionalization of the matrix A:
    U, A, V = bidiag(A)

    return 0


def svd_step(B):
    """
    Computes a single step of the Golub-Kahan SVD iteration. 
    """
    # trailing eigenvalue
    mu = T[n,n] - (T[n, n-1]**2)/(d + np.sign(d) * np.sqrt(d**2 + T[n,n-1]**2))
    return 0

    
def cs(Q1, Q2):
    """
    Computes a cosine-sine decomposition
    """
    m, p = Q1.shape
    n, pb = Q2.shape

    if p != pb:
        raise(ValueError, 'cs(): Number of columns in Q1 must be equivalent to number of columns in Q2')
    
    if m < n:
        V, U, Z, S, C = cs(Q2, Q1)
        j = range(p, 0, -1)
        C = C[:, j]
        S = S[:, j]
        Z = Z[:, j]
        m = np.min([m,p])
        n = np.min([n, p])
        i = range(m,0, -1)
        C[0:m, :] = C[i,:]
        U[:, 0:m] = U[:, i]
        i = range(n,0, -1)
        S[0:n, :] = S[i, :]
        V[:, 0:n] = V[:, i]
        return U, V, Z, C, S
    
    # Compute the svd
    U, C, Z = np.linalg.svd(Q1)
    q = np.min([m,p])
    i = range(0, q, 1)
    j = range(q, 0, -1)
    C[i,i] = C[j,j]
    U[:,i] = U[:,j]
    Z[:,i] = Z[:,j]
    S = Q2 * Z

    if q == 1:
        k = 0
    elif m < p:
        k = n
    else:
        entries = np.diag(C)
        k = np.max([np.nonzero(entries <= 2)])
    
    V, R = qr_Householder(S[:, 0:k])
    S = V.T * S
    r = np.min([k,m])
    S[:, 0:r] = np.diag(S[:, 0:r])
    if m == 1 and p > 1:
        S[0,0] = 0

    if k < np.min([n,p]):
        r = np.min([n,p])
        i = range(k+1, n, 1)
        j = range(k+1, r, 1)

        # Compute svd!
        [UT, ST, VT] = np.linalg.svd(S[i, j])
        
        if k > 0:
            S[1:k, j] = ST
        
        C[:, j] = C[:, j] * VT
        V[:, i] = V[:, i] * UT
        Z[:, j] = Z[:, j] * VT
        i = range(k,q,1)
        Q, R = qr(C[i,j])
        C[i,j] = np.triu(np.tril(R))
        U[:,i] = U[:, i] * Q
        
    if m < p:

       return 0 


def implicitSymmetricQR(T):
    """
    Computes an implicit step for a symmetric QR factorization
    see Golub and Van Loan (4th Ed., page 462)
    """
    # Preliminary parameters
    m, n = T.shape # but because its symmetric m = n!
    d = ( T[n-1, n-1] - T[n, n] )/(2.0)
    mu = ( T[n, n] - T[n, n-1]**2 )/( d + np.sign(d) * np.sqrt(d**2 + T[n, n-1]**2 ) )
    x = T[0, 0] - mu
    z = T[1, 0]

    # For loop with Givens rotation
    for k in range(0, n-1):
        [c, s] = givens(x, z)
        G = np.mat(np.eye(n), dtype='float64')
        G[k,k] = c
        G[k, k+1] = -s 
        G[k+1, k] = s
        G[k+1, k+1] = c
        T = G.T * T * G
        if k < n - 1:
            x = T[k+1, k] 
            z = T[k+2, k]
    
    return T
        

def givens(a, b):
    """ 
    Computes a Givens rotation; computes c = cos(theta), s = sin(theta) so that
        [ c   s]' [ a ] = [ r ]
        [-s   c]  [ b ]   [ 0 ]
    for scalars a and b.

    """
    #G = np.mat(np.eye(2), dtype='float64')
    if b == 0:
        c = 1
        s = 1
    else:
        if np.abs(b) > np.abs(a):
            tau = -(1.0 * a)/(b * 1.0)
            s = 1.0/np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = (-1.0 * b)/(a * 1.0)
            c = 1.0/np.sqrt(1 + tau**2)
            s = c * tau
            
    # Place values of c and s into G
    #G[0,0] = c
    #G[0,1] = s
    #G[1,0] = -1.0 * s
    #G[1,1] = c

    return c, s

def bidiag(A):
    """
    Computes the bidiagonalization of the m-by-n matrix A := U B V', where
    m >=n. Here U is an m-by-n orthogonal matrix, B is a n-by-n bi-diagonal matrix
    and V is a n-by-n orthogonal matrix.

    >> clean up comments below!

    :param numpy matrix A: an m-by-n matrix
    :param numpy matrix b: an m-by-1 matrix
    :param numpy ndarray C: an k-by-n matrix
    :param numpy ndarray d: an k-by-1 matrix
    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray

    ** Notes **
    Uses the algorithm of Golub and Kahan (1965) and requires 4mn^2 - 4n^3/3 flops.
    """
    m, n = A.shape
    if m < n :
        raise(ValueError, 'bidiag(A): Only valid for an m-by-n matrix A, wher m>=n.')

    # Allocate space
    U = np.mat(np.vstack( [np.eye(n), np.zeros((m-n,n)) ] ), dtype='float64')
    V = np.mat(np.eye(n,n), dtype='float64')
    G = np.mat(np.eye(m,n), dtype='float64') # For storing U and V Householder vectors

    # For loop for computing bi-diagional matrix A
    for j in range(0, n):
        v, beta = house(A[j:m, j])
        A[j:m, j:n] = (np.identity(m-j) - np.multiply(beta,  v * v.T )) * A[j:m, j:n]
        G[j+1:m, j] = v[1:m-j+1]

        if j <= n - 2 : 
            v, beta = house(A[j,j+1:n].T)
            A[j:m, j+1:n] = A[j:m, j+1:n] * (np.identity(n-j-1) -  np.multiply(beta,  v * v.T ) )
            G[j,j+2:n] = v[1 : n-j].T
   
    # Unpack U 
    for j in range(n-1, -1, -1):
        if j >= m - 1:
            if not G[j+1:m, j]:
                beta = 2.0
                U[j:m, j:n] = U[j:m, j:m] - ( np.multiply( beta,  U[j:m, j:m] ) )
        else:    
            v = np.mat(np.vstack([1.0, G[j+1:m, j] ]), dtype='float64')
            beta =  2.0/(1.0 + np.linalg.norm(G[j+1:m, j], 2)**2)
            U[j:m, j:n] = U[j:m, j:n] - ( ( np.multiply( beta,  v) ) * (v.T  * U[j:m, j:n] ) )

    # Unpack V 
    for j in range(n-2, -1, -1):
        if j == n-2:
            beta = 2.0
            V[j+1:n, j:n] = V[j+1:n, j:n] - ( np.multiply( beta,  V[j+1:n, j:n] ) )
        else:
            v = np.mat( np.vstack([1.0, G[j, j+2:n].T  ]), dtype='float64')
            beta =  2.0/(1.0 + np.linalg.norm(G[j,j+2:n], 2)**2)
            V[j+1:n, j:n] = V[j+1:n, j:n] - ( ( np.multiply( beta,  v) ) * (v.T  * V[j+1:n, j:n] ) )

    # Remove trailing zeros from A
    A = A[0:n, 0:n]
    return U, A, V

def solveCLSQ(A,b,C,d):
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
    # Size of matrices!
    A = np.mat(A)
    C = np.mat(C)
    b = np.mat(b)
    d = np.mat(d)
    m, n = A.shape
    p, q = b.shape
    k, l = C.shape
    s, t = d.shape
    
    # Check that the number of elements in b are equivalent to the number of rows in A
    if m != p:
        raise(ValueError, 'solveCLSQ(): mismatch in sizes of A and b')
    elif k != s:
        raise(ValueError, 'solveCLSQ(): mismatch in sizes of C and d') 

    Q , R = qr_Householder(C.T)
    R = R[0:k, 0:k]
    u = np.linalg.inv(R.T) * d
    A_hat = A * Q
    z, w = A_hat.shape
    Ahat_1 = A_hat[:, 0:len(u)]
    Ahat_2 = A_hat[:, len(u) : w]
    r = b - Ahat_1 * u
    v = solveLSQ(Ahat_2, r)
    x = Q * np.mat(np.vstack([u, v]) , dtype='float64')
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
    A = np.mat(A)
    b = np.mat(b)
    Q, R = qr_MGS(A)
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

def qr_MGS(A):
    """
    Returns the thin QR factorization via the Modified Gram Schmidt Method

    :param numpy matrix A: Matrix input for QR factorization
    :param string thin: Set thin to 'yes' to compute a thin QR factorization. The default is a regular QR factorization, i.e., the string is set to 'no'.
    :return: Q, the m x n orthogonal matrix
    :rtype: numpy matrix
    :return: R, the n x n upper triangular matrix
    :rtype: numpy matrix

    **Sample declaration**
    :: 
        >> Q, R = qr(A)
    """
    A = np.matrix(A)
    m , n = A.shape
    Q = np.mat(np.eye(m,n), dtype='float64')
    R = np.mat(np.zeros((n, n)), dtype='float64')

    
    # Min and maximum values
    u = np.min([m,n]) 
    h = np.max([m,n])

    # Outer for loop for MGS QR factorization
    for k in range(0, u):

        # Re-orthogonalization
        if k != 0:
            for i in range(0, k-1):
                alpha = (Q[0:m,i]).T * A[0:m,k]
                R[i,k] = R[i,k] + alpha[0,0]
                A[0:m,k] = A[0:m,k] - alpha[0,0]*Q[0:m,i]

        # Normalization
        R[k,k] = np.linalg.norm(A[0:m, k], 2)
        const = R[k,k]
        Q[0:m, k] = A[0:m, k] * 1.0/(const)

        # Orthogonalization
        if k != n:
            for j in range(k+1, n):
                R[k,j] = (Q[0:m,k]).T * A[0:m,j];
                A[0:m,j] = A[0:m,j] - R[k,j]* Q[0:m,k];          
            
    return Q, R

# QR factorization via the method of Householder
def qr_Householder(A):
    """
    Returns the Householder QR factorization of a matrix A 

    :param numpy matrix A: Matrix input for QR factorization
    :param boolean thin: Set thin to 1 to compute a thin QR factorization. The default is a regular QR factorization.
    :return: Q, the m x m orthogonal matrix
    :rtype: numpy matrix
    :return: R, the m x n upper triangular matrix
    :rtype: numpy matrix

    **Sample declaration**
    :: 
        >> Q, R = qr_Householder(A)
    """
    A = np.mat(A)
    m, n = A.shape
    k = np.min([m,n])
    Q = np.mat(np.eye(m,m), dtype='float64')
    R = np.mat(np.eye(m,n), dtype='float64')

    for j in range(0, k):
        v, beta = house(A[j:m, j])
        A[j:m, j:n] = (np.identity(m-j) - np.multiply(beta,  v * v.T )) * A[j:m, j:n]
        # The section below ensures that the lower triangular portion of A stores the Householder
        # vectors that will be used when computing Q!
        if j < m :
            A[j+1:m, j] = v[2-1:m-j+1] 

    R = np.triu(A)

    # Computing Q using backward accumulation of lower triangular part of A!
    for j in range(n-1, -1, -1):
        if j >= m - 1:
            if not A[j+1:m, j]:
                beta = 2.0
                Q[j:m, j:m] = Q[j:m, j:m] - ( np.multiply( beta,  Q[j:m, j:m] ) )
        else:
            v = np.mat(np.vstack([1.0, A[j+1:m, j] ]), dtype='float64')
            beta =  2.0/(1.0 + np.linalg.norm(A[j+1:m, j], 2)**2)
            Q[j:m, j:m] = Q[j:m, j:m] - ( ( np.multiply( beta,  v) ) * (v.T  * Q[j:m, j:m] ) )


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