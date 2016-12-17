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
def cs(A, B):
    """
    Computes the CS decomposition!
    """


def gsvd(A,B):
    """
    Computes a generalized singular value decomposition
    """

def svd(A):
    """
    Computes the singular value decomposition of a matrix!
    """

def bidiag(A):
    """
    Computes a bidiagionalization of a matrix

    ** Notes **
    Uses the algorithm of Golub and Kahan (1965) and requires 4mn^2 - 4n^3/3 flops.
    """
    m, n = A.shape
    G = np.mat(np.eye(m,n), dtype='float64') # For storing U and V Householder vectors
    for j in range(0, n):
        v, beta = house(A[j:m, j])
        A[j:m, j:n] = (np.identity(m-j) - np.multiply(beta,  v * v.T )) * A[j:m, j:n]
        G[j+1:m, j] = v[1:m-j+1]
        #print 'Pass'

        if j <= n - 2 : 
            v, beta = house(A[j,j+1:n].T)
            A[j:m, j+1:n] = A[j:m, j+1:n] * (np.identity(n-j-1) -  np.multiply(beta,  v * v.T ) )
            G[j,j+2:n] = v[1 : n-j].T
            
    # Unpack U and V -- using backward accumulation!?
    for j in range(n-1, -1, -1):
        if j >= m - 1:
            if not A[j+1:m, j]:
                beta = 2.0
                Q[j:m, j:m] = Q[j:m, j:m] - ( np.multiply( beta,  Q[j:m, j:m] ) )
        else:
            v = np.mat(np.vstack([1.0, A[j+1:m, j] ]), dtype='float64')
            beta =  2.0/(1.0 + np.linalg.norm(A[j+1:m, j], 2)**2)
            Q[j:m, j:m] = Q[j:m, j:m] - ( ( np.multiply( beta,  v) ) * (v.T  * Q[j:m, j:m] ) )

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
    R = R[0:m, 0:m]
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
    Q, R = qr_Householder(A)
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

def main2():
    
    A = np.mat(np.random.rand(16,5), dtype='float64')
    print bidiag(A)

def main():
    

    A = np.mat([ [ 0.22497615,  0.37419627,  0.44432189, 0.46035715 , 0.43012769 , 0.36028308, 0.25875463,  0.13507237],
    [0.42584257,-0.1352979,  -0.42804599,  0.29262113,  0.32420127, -0.4117993, -0.17415016,  0.47590828],
    [0.33345242, -0.46011993,  0.33703336, -0.06093221, -0.24280944,  0.4460431, -0.46503932,  0.29261317],
    [0.39604712, -0.36050143, -0.07591575,  0.44579303, -0.3884936,  -0.04077974, 0.43180574, -0.41187899]], dtype='float64')

    C = np.mat([[ 0.0 ,        0.0    ,     0.0       ,  0.0],
    [1.7321,    1.7321 ,   1.7321,    1.7321],
    [6.4418 ,  -1.2305 ,  -5.3442,   -3.5254],
    [14.3299 ,  -3.3009 ,   8.6254,    1.5117],
    [24.8842 ,   3.8032 ,  -8.6204,    4.2044],
    [36.9864 ,   3.4371 ,   3.5673,   -7.8650],
    [49.0574 ,  -6.9930,    5.6356,    3.8218],
    [59.2517 ,  -1.6971  ,-15.3091,    6.0407]], dtype='float64')
    C = C.T

    b = np.mat([[ 0.58773975],
    [0.35447403],
    [0.15033013],
    [0.2341591]], dtype='float64')
 
    d = np.mat([[ 2.6124536], 
    [0.83240628],
    [0.45082931],
    [0.5912405] ], dtype='float64')

    x = solveCLSQ(A, b, C, d)


main2()