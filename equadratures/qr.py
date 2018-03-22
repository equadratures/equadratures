#!/usr/bin/env python
"""Utilities with QR factorization."""
import numpy as np
from scipy.optimize import minimize

def implicitSymmetricQR(T):
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



def qr_Givens(A):
    """
    Returns the Givens QR factorization of a matrix A

    :param numpy matrix A: Matrix input for QR factorization
    :return: Q, the m x m orthogonal matrix
    :rtype: numpy matrix
    :return: R, the m x n upper triangular matrix
    :rtype: numpy matrix

    **Sample declaration**
    ::
        >> Q, R = qr_Givens(A)
    """

    # Declarations!
    A = np.mat(A)
    m, n = A.shape
    #Q_trans = np.mat(np.eye(m,m), dtype='float64')
    G = np.mat( np.eye(2,2), dtype='float64')
    G_store = np.mat( np.eye(m, m), dtype='float64')

    # Introduce zeros by rows
    for i in range(1, m):
        for j in range(0, i):
            if j < n:
                c, s = givens(A[j,j], A[i,j])
                G[0,0] = c
                G[0,1] = s
                G[1,0] = -1.0 * s
                G[1,1] = c
                A[[j, i], j:n ] = G.T * A[[j, i], j:n ]

                # Collect the individual Givens rotations!
                G_local = np.mat( np.eye(m, m), dtype='float64')
                G_local[i, i] = c
                G_local[i, j] = s
                G_local[j, i] = -1.0 * s
                G_local[j, j] = c
                G_store = G_local * G_store

    R = A

    return G_store.T, R

def bidiag(A):
    """
    Computes the bidiagonalization of the m-by-n matrix A := U B V', where
    m >=n. Here U is an m-by-n orthogonal matrix, B is a n-by-n bi-diagonal matrix
    and V is a n-by-n orthogonal matrix.

    :param numpy matrix A: an m-by-n matrix
    :return: U, the left orthogonal matrix
    :rtype: numpy matrix
    :return: B, the bidiagonal matrix
    :rtype: numpy matrix
    :return: V, the right orthogonal matrix
    :rtype: numpy matrix

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

def solveCLSQ(A,b,C,d, technique=None):
    """
    Solves the direct, constraint least squares problem ||Ax-b||_2 subject to Cx=d.

    :param numpy matrix A: an m-by-n matrix
    :param numpy matrix b: an m-by-1 matrix
    :param numpy ndarray C: an k-by-n matrix
    :param numpy ndarray d: an k-by-1 matrix
    :param string technique: The technique for solving the least squares problem. Options include, weighted, constrainedDE, constrainedNS.
    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray


    method options: 'equality', 'weighted', 'inequality',...
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

    # Stacked least squares
    if technique is 'weighted' or technique is None:
        x, cond = solveLSQ(np.mat(np.vstack([A, C])), np.mat(np.vstack([b, d])))
    elif technique is 'constrainedDE':
        x, cond = directElimination(C, d, A, b)
    elif technique is 'constrainedNS':
        x, cond = nullSpaceMethod(C, d, A, b)
    return x, cond


def nullSpaceMethod(A, b, C, d):
    """
    Solves the constrained least squares problem min ||Ax-b||_2 subject to Cx=d via the null space method.
    :param numpy ndarray A: an m-by-n A matrix
    :param numpy ndarray b: an m-by-1 b matrix

    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray
    :return: cond, the condition number of the final matrix on which least squares is performed
    :rtype: float
    """
    m, n = A.shape
    p, n = C.shape

    Q, R = qr_Householder(C.T)
    Q1 = Q[0:n, 0:p]
    Q2 = Q[0:n, p:n]

    # Lower triangular matrix!
    L = R.T
    L = L[0:p, 0:p]
    y1, not_required = solveLSQ(L, d)
    c = b - (A * Q1) * y1
    AQ2 = A * Q2
    y2, not_required = solveLSQ(AQ2 , c)
    x = (Q1 * y1) + (Q2 * y2)
    cond = np.linalg.cond(AQ2)
    return x, cond

def directElimination(A, b, C, d):
    """
    Solves the constrained least squares problem min ||Ax-b||_2 subject to Cx=d via the direct elimination method.
    :param numpy ndarray A: an m-by-n A matrix
    :param numpy ndarray b: an m-by-1 b matrix

    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray
    :return: cond, the condition number of the final matrix on which least squares is performed
    :rtype: float
    """
    Q, R, pvec = qr_MGS(C, pivoting=True)
    m1, n1 = R.shape
    P = permvec2mat(pvec)
    r = np.linalg.matrix_rank(C)
    R_11 = R[0:r, 0:r]
    R_12 = R[0:r, r:n1]
    d_tilde = Q.T * d
    d1_tilde = d_tilde[0 : r]
    d2_tilde = d_tilde[r: m1]
    A_tilde = A * P
    A1_tilde = A_tilde[:, 0 : r]
    A2_tilde = A_tilde[:, r : m1]
    A2_hat = A2_tilde - A1_tilde * np.linalg.inv(R_11) * R_12
    b_hat = b - A1_tilde * np.linalg.inv(R_11) * d1_tilde
    x2_tilde , cond_not_used = solveLSQ(A2_hat, b_hat)
    x1_tilde = np.linalg.inv(R_11) * (d1_tilde - R_12 * x2_tilde)
    x_tilde = np.mat( np.vstack([x1_tilde, x2_tilde]) , dtype='float64')
    cond = np.linalg.cond(A2_hat)
    return P * x_tilde, cond

def solveLSQ(A, b):
    """
    Solves the direct least squares problem ||Ax-b||_2 using the method of QR factorization

    :param numpy ndarray A: an m-by-n A matrix
    :param numpy ndarray b: an m-by-1 b matrix

    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray
    :return: cond, the condition number of the matrix A
    :rtype: float

    """
    # Direct methods!
    A = np.mat(A)
    b = np.mat(b)
    Q, R = qr_MGS(A)
    x = np.linalg.inv(R) * Q.T * b
    x = np.array(x)
    return x, np.linalg.cond(A)


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

def qr_MGS(A, pivoting=None):
    """
    Returns the thin QR factorization via the Modified Gram Schmidt Method

    :param numpy matrix A: Matrix input for QR factorization
    :param sting pivoting: Set pivoting to yes to compute a QR with column pivoting factorization.
    :return: Q, the m x n orthogonal matrix
    :rtype: numpy matrix
    :return: R, the n x n upper triangular matrix
    :rtype: numpy matrix

    **Sample declaration**
    ::
        >> Q, R = qr_MGS(A)
        >> Q, R, p = qr_MGS(A, pivoting='yes')
    """
    if pivoting is not None:
        return qr_MGS_Pivoting(A)

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
        Q[:,k] = np.array(A[:,k] * 1.0/R[k,k])

        # Orthogonalization
        if k != n:
            for j in range(k+1, n):
                R[k,j] = (Q[0:m,k]).T * A[0:m,j];
                A[0:m,j] = A[0:m,j] - R[k,j]* Q[0:m,k];

    return Q, R

def qr_MGS_Pivoting(A):

    # Determine the size of
    A = np.matrix(A)
    m , n = A.shape
    u = np.min([m, n])
    h = np.max([m, n])
    Q = np.mat(np.eye(m,n), dtype='float64')
    R = np.mat(np.zeros((n, n)), dtype='float64')

    # Initialize!
    column_norms = np.zeros((n))
    pivots = np.linspace(0,n-1,n, dtype='int16')

    # Compute the column norms
    for j in range(0,n):
        column_norms[j] = np.linalg.norm(np.array(A[:,j]), 2)**2


    # Now loop!
    for k in range(0, u):

        #----------------------------------------------
        # Step 0: Column norm sorting
        #----------------------------------------------
        # Find the "j*" column index with the highest
        # column norm
        j_star = np.argmax(column_norms[k:n])
        r_star = j_star + k


        # If j_star = k, skip to step 1, else swap columns!
        if k != j_star:

            # Swap columns in A:
            A[0:m, [r_star, k]] = A[0:m, [k, r_star]]
            R[0:m, [r_star, k]] = R[0:m, [k, r_star]]
            temp = pivots[r_star]
            pivots[r_star] = pivots[k]
            pivots[k] = temp
            del temp
        #-----------------------------------------------
        # Step 1: Reorthogonalization
        #-----------------------------------------------
        if k != 0:
            for i in range(0,k):
                alpha = np.dot(Q[:,i].T , A[:,k] )
                R[i,k] = R[i,k] + alpha
                A[:,k] = np.array(A[:,k] - alpha[0,0] * Q[:,i])

        #----------------------------------------------
        # Step 2: Normalization
        #----------------------------------------------
        R[k,k] = np.linalg.norm(np.array(A[:,k]), 2)
        if np.abs(R[k,k]) >= 1e-15:
            Q[:,k] = np.array(A[:,k] * 1.0/R[k,k])
        else:
            Q[:,k] = np.array(A[:,k])
        #----------------------------------------------
        # Step 3: Orthogonalization
        #----------------------------------------------
        if k != n:
            for j in range(k,n):
                R[k,j] = np.dot(Q[:,k].T , A[:,j] )
                A[:,j] = np.array(A[:,j] - np.mat(R[k,j] * Q[:,k]))

                # Now re-compute column norms
                column_norms[j] = np.linalg.norm(np.array(A[:,j]), 2)**2


    return Q, R, pivots

def qr_Householder(A):
    """
    Returns the Householder QR factorization of a matrix A

    :param numpy matrix A: Matrix input for QR factorization
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

def permvec2mat(vec):
    """
    Converts a permutation vector into a permutation matrix

    :param numpy matrix p: A 1 x k vector of permutation indices (all integers)
    :return: P, the k x k permutation matrix
    :rtype: numpy matrix

    """
    n = len(vec)
    P = np.mat(np.zeros((n,n)), dtype='float64')
    counter = 0
    for i in range(0, n):
        for j in range(0, n):
            if j == vec[counter]:
                P[i,j] = 1
                counter = counter + 1
                break
    return P.T

# Private functions below!
def givens(a, b):

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

    return c, s

def rowNormalize(A):
    rows, cols = A.shape
    row_norms = np.mat(np.zeros((rows, 1)), dtype='float64')
    Normalization = np.mat(np.eye(rows), dtype='float64')
    for i in range(0, rows):
        temp = 0.0
        for j in range(0, cols):
            row_norms[i] = temp + A[i,j]**2
            temp = row_norms[i]
        row_norms[i] = (row_norms[i] * 1.0/np.float64(cols))**(-1)
        Normalization[i,i] = row_norms[i]
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization
