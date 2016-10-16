#!/usr/bin/env python
"""Utilities with QR factorization"""
import numpy as np
from utils import error_function

# Solve the weighted least squares problem, using the method of row     
def solve_weightedLSQ(A, b, C, d, alpha):
    """
    Solve the constraint least squares problem || [A; C]x-[b; d]||_2

    :param ndarray A: m-by-n matrix of linear equations
    :param ndarray b: m-by-1 matrix of right-hand side of the linear equations given by A
    :param ndarray C: m-by-n matrix of linear equations
    :param ndarray d: m-by-1 matrix of right-hand side of the linear equations given by C

    :return x: n-by-1 matrix that contains coefficients to the weighted least squares problem.
    :rtype: ndarray

    **Notes**

    We utilize the 
    """
    x = 0
    return x
    
# Solve the constrained least squares problem, using method of direct elimination
def solve_constrainedLSQ(A,b,C,d):
    """
    Solve the constraint least squares problem ||Ax-b||_2 s.t Cx = d

    :param ndarray A: m-by-n matrix of linear equations
    :param ndarray b: m-by-1 matrix of right-hand side of the linear equations given by A
    :param ndarray C: m-by-n matrix of linear equations
    :param ndarray d: m-by-1 matrix of right-hand side of the linear equations given by C

    :return x: n-by-1 matrix that contains coefficients to the weighted least squares problem.
    :rtype: ndarray

    **Notes**

    We utilize the 
    """

    # Preliminaries
    temp , R = qr_householder(C.T, 1) # Thin QR factorization on C'
    Q , temp = qr_householder(C.T) # Regular QR 
    u = np.linalg.inv(R) * d
    Ahat = A * Q
    m, n = Ahat.shape
    
    # Now split A
    Ahat_1 = A_hat[:, 0:len(u)-1]
    Ahat_2 = A_hat[:, len(u) : n]
    r = b - Ahat_1 * u
    
    # Solve the least squares problem
    v = solveLSQ(Ahat_2, r)
    x = Q * [u, v] #----> Need to somehow concatenate this!!!
    return x

# Solve the least squares problem (done wiht QR factorization!)
# Perhaps also add LU and SVD techniques to solve least squares!
def solveLSQ(A, b):
    
    # Direct methods!
    Q, R = qr_householder(A, 1)
    x = np.linalg.inv(R) * Q.T * b
    x = np.array(x)
    return x

# Householder reflection
def house(vec):
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
def qr_householder(A, thin=None):
    """
    Returns the Householder QR factorization of a matrix A

    :param numpy ndarray A: A matrix
    :param boolean thin: Use a value of 1 for a thin QR 
      
    :return: x, the coefficients of the least squares problem.
    :rtype: ndarray

        **Sample declaration**
        :: 
            >> qr_householder(A)
        """
    


    if thin is None:
        thin = 0
    else:
        thin = 1
    A = np.mat(A)
    m, n = A.shape # size of A

    for j in range(0, n):
        v, beta = house(A[j:m, j])
        K = np.multiply(beta,  v * (v.T) )
        A[j:m, j:n] = (np.identity(m-j) - K ) * A[j:m, j:n]
        if j < m :
            A[j+1:m, j] = v[1:m-j+1]
    
    R = np.triu(A)

    # Computation of Q using backward accumulation
    k = min([m,n])
    Q = np.identity(m)
    for j in range(k-1, -1, -1):
        if k == m and j == k-1 :
            v = [1]
            beta = 2
            Q[j:m, j:m] = -1
        else:
            v = np.insert(A[(j+1):m, j], 0, 1)
            v = np.mat(v)
            beta = 2/(1 + np.linalg.norm( A[(j+1):m, j] , 2)**2 )
            G = np.multiply(beta,  v.T * v )
            Q[j:m, j:m] = Q[j:m, j:m] - ( G * Q[j:m, j:m] )

    # For making it thin!
    if thin == 1:
        Q = Q[0:m, 0:n]
        R = R[0:n, 0:n]
    
    return Q, R

# Modified Gram Schmit QR column pivoting
def mgs_pivoting(A):

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

def norms(vector, integer):

    # l1 norm
    if integer == 1:
        return np.sum(np.abs(vector))
    elif integer == 2:
        # l2 norm
        return np.sqrt(np.sum(vector**2))
    else:
        error_function('Norm: Either using 1 or 2 as the second argument for the norm')

def main():
    
    #-----------------------------------------------------------------
    # Test 1: Check if Householder vector works!
    #-----------------------------------------------------------------
    x = [4., 2., 1., -2., 4., 1.]
    x = np.mat(x)
    v, beta = house(x.T)

    # Know solution
    real_v = [1.0000, -0.80621082,  -0.4031054,  0.80621082, -1.61242164,-0.4031054]
    real_v = np.mat(real_v)
    real_v = real_v.T

    if np.linalg.norm(real_v - v, 2) < 1e-07:
        print 'Success!'
    else:
        error_function('ERROR: Householder vector incorrect!')
    
    #-----------------------------------------------------------------
    # Test 2: Check if QR factorization works! (m>=n)
    #-----------------------------------------------------------------
    A =  [ [0.8147,    0.0975,    0.1576,    0.1419], [0.9058,    0.2785,    0.9706,    0.4218], [0.1270,    0.5469 ,   0.9572 ,   0.9157], [0.9134  ,  0.9575 ,   0.4854   , 0.7922], [0.6324 ,   0.9649,    0.8003 ,   0.9595]]
    Q, R = qr_householder(A)
    Q1, R1 = qr_householder(A,1)
    if np.linalg.norm(A - ( Q * R), 2) < 1e-15 and np.linalg.norm(A - ( Q1 * R1), 2) < 1e-15 :
        print 'Success!'
    else:
        error_function('ERROR: QR Householder not working!')
    
    #-----------------------------------------------------------------
    # Test 3: Check if QR factorization works! (m<n)
    #-----------------------------------------------------------------
    A = [[ 0.549723608291140 ,  0.380445846975357 ,  0.779167230102011 ,  0.011902069501241 ,  0.528533135506213 ,  0.689214503140008 ,  0.913337361501670 ,  0.078175528753184   ,0.774910464711502], 
    [0.917193663829810  , 0.567821640725221  , 0.934010684229183  , 0.337122644398882  , 0.165648729499781  , 0.748151592823709  , 0.152378018969223  , 0.442678269775446  , 0.817303220653433],
    [ 0.285839018820374 ,  0.075854289563064 ,  0.129906208473730 ,  0.162182308193243 ,  0.601981941401637 ,  0.450541598502498 ,  0.825816977489547 ,  0.106652770180584 ,  0.868694705363510],
    [0.757200229110721  , 0.053950118666607  , 0.568823660872193  , 0.794284540683907  , 0.262971284540144  , 0.083821377996933  , 0.538342435260057  , 0.961898080855054  , 0.084435845510910],
    [ 0.753729094278495 ,  0.530797553008973 ,  0.469390641058206 ,  0.311215042044805 ,  0.654079098476782 ,  0.228976968716819 ,  0.996134716626885 ,  0.004634224134067 ,  0.399782649098896]]

    Q, R = qr_householder(A)
    print Q 
    print '\n'
    print R



    #residual = real_v - v
    #print residual

main()
"""
    QR Householder with Pivoting!

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
   
