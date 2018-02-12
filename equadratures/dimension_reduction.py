"""Dimension Reduction Functionalities"""
import numpy as np
from parameter import Parameter
from poly import Poly

def computeActiveSubspaces(PolynomialObject, samples=None):
    d = PolynomialObject.dimensions
    if samples is  None:
        M = 300
        X = np.zeros((M, d))
        for j in range(0, d):
            X[:, j] =  np.reshape(PolynomialObject.parameters[j].getSamples(M), M)
    else:
        X = samples
        M, _ = X.shape
        X = samples

    # Gradient matrix!
    polygrad = PolynomialObject.getPolyGradFit(xvalue=X)
    weights = np.ones((M, 1)) / M
    R = polygrad.transpose() * weights
    C = np.dot(polygrad, R )

    # Compute eigendecomposition!
    e, W = np.linalg.eigh(C)
    idx = e.argsort()[::-1]
    eigs = e[idx]
    eigVecs = W[:, idx]
    return eigs, eigVecs

def linearModel(training_X, training_Y):
    # Linear OLS model goes here!
    return 0