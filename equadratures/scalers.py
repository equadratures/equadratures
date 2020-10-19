"""Data Scaling classes"""
import numpy as np

class Scaler_minmax(object):
    """ 
    Scale the data to have a min/max of -1 to 1.
    """
    def __init__(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        self._fit(X)

    def _fit(self,X):
        self.Xmin = np.min(X,axis=0)
        self.Xmax = np.max(X,axis=0)

    def transform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        Xtrans = 2.0 * ( (X[:,:]-self.Xmin)/(self.Xmax - self.Xmin) ) - 1.0
        return Xtrans 

    def untransform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        Xuntrans = 0.5*(X[:,:]+1)*(self.Xmax - self.Xmin) + self.Xmin
        return Xuntrans

class Scaler_meanvar(object):
    """ 
    Scale the data to have a mean of 0 and variance of 1.
    """
    def __init__(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        self._fit(X)

    def _fit(self,X):
        self.Xmean = np.mean(X,axis=0)
        self.Xstd  = np.std(X,axis=0)

    def transform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xtrans = (X[:,:]-self.Xmean)/(self.Xstd+eps)
        return Xtrans 

    def untransform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xuntrans = X[:,:]*(self.Xstd+eps) + self.Xmean
        return Xuntrans

