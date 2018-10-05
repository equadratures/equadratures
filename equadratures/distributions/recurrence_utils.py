"""Recurrence coefficients class."""
import numpy as np 
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

def jacobi_recurrence_coefficients(a, b, lower, upper, order):
    """
    Returns the Jacobi recurrence coefficients.
        
    :param double a:
        First shape parameter.
    :param double b:
        Second shape parameter.
    :param double lower:
        Lower bound of the Jacobi parameter.
    :param double upper:
        Upper bound of the Jacobi parameter.
    :param int order:
        Order of the recurrence coefficients requested.
    :return:
        (order+1)-by-2 numpy array of the Jacobi recurrence coefficients.
    """
    nn = int(order) + 1
    a0 = 1.*(b - a)/(a + b + 2.)
    ab = np.zeros((nn,2))
    b2a2 = b**2 - a**2
    s = (upper - lower)/2.
    other = lower + (upper - lower)/2.
    if nn > 0:
        ab[0,0] = s*a0 + other
        ab[0,1] = 1.0
    for i in range(1, nn):
        k = i + 1
        ab[i, 0] = s * b2a2/((2.*(k-1.) + a + b) * (2.*k + a + b)) + other
        if i == 1:
            ab[i, 1] = ( (upper - lower)**2 * (k - 1.) * (k - 1. + a) * (k - 1. + b) )/( (2. * (k - 1.) + a + b)**2 * (2. * (k - 1.) + a + b + 1. )  )
        else:
            ab[i, 1] = ( (upper - lower)**2 * (k - 1.) * (k - 1. + a) * (k - 1. + b) * (k - 1. + a + b))/( (2. * (k - 1.) + a + b)**2 * (2. * (k-1.) + a + b + 1.)* (2. * (k-1.) + a + b - 1.) )
    return ab

def hermite_recurrence_coefficients(param_A, param_B, order):
    """
    Returns the Hermite recurrence coefficients.
        
    :param double param_A:
        First shape parameter.
    :param double param_B:
        Second shape parameter.
    :param int order:
        Order of the recurrence coefficients requested.
    :return:
        (order+1)-by-2 numpy array of the Hermite recurrence coefficients.
    """
    ab = np.zeros((order,2))
    sigma2 = param_B
    
    if order == 1:
        ab[0,0] = 0
        ab[0,1] = gamma(param_A + 0.5)
        return ab
    
    # Adapted from Walter Gatuschi
    N = order - 1
    n = range(1,N+1)
    nh = [ k / 2.0 for k in n]
    for i in range(0,N,2):
        nh[i] = nh[i] + sigma2
    
    # Now fill in the entries of "ab"
    for i in range(0,order):
        if i == 0:
            ab[i,1] = gamma(sigma2 + 0.5)
        else:
            ab[i,1] = nh[i-1]
    ab[0,1] = gamma(param_A + 0.5)#2.0

    return ab

def custom_recurrence_coefficients(x, w, order):
    """
    Returns the custom recurrence coefficients.
        
    :param array x:
        Equidistant values of the support of the distribution.
    :param array w:
        Probability density function weights associated with the distribution.
    :param int order:
        Order of the recurrence coefficients requested.
    :return:
        (order+1)-by-2 numpy array of the Hermite recurrence coefficients.
    """
    
    # Allocate memory for recurrence coefficients
    order = int(order)+1
    w = w / np.sum(w)
    ab = np.zeros((order+1,2))
    
    # Negate "zero" components
    nonzero_indices = []
    for i in range(0, len(x)):
        if w[i] != 0:
            nonzero_indices.append(i)

    ncap = len(nonzero_indices)
    x = x[nonzero_indices] # only keep entries at the non-zero indices!
    w = w[nonzero_indices]
    s = np.sum(w)
    temp = w/s
    ab[0,0] = np.dot(x, temp.T)
    ab[0,1] = s
        
    if order == 1:
        return ab

    p1 = np.zeros((1, ncap))
    p2 = np.ones((1, ncap))

    for j in range(0, order):
        p0 = p1
        p1 = p2
        p2 = ( x - ab[j,0] ) * p1 - ab[j,1] * p0
        p2_squared = p2**2
        s1 = np.dot(w, p2_squared.T)
        inner = w * p2_squared
        s2 = np.dot(x, inner.T)
        ab[j+1,0] = s2/s1
        ab[j+1,1] = s1/s
        s = s1
    return ab
