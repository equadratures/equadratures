"""Data Scaling classes"""
import numpy as np

class scaler_minmax(object):
    """ 
    Scale the data to have a min/max of -1 to 1.
    """
    def __init__(self):
        self.fitted = False

    def fit(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        self.Xmin = np.min(X,axis=0)
        self.Xmax = np.max(X,axis=0)
        self.fitted = True

    def transform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted: self.fit(X)
        Xtrans = 2.0 * ( (X[:,:]-self.Xmin)/(self.Xmax - self.Xmin) ) - 1.0
        return Xtrans 

    def untransform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted:
            raise Exception('scaler has not been fitted')
        Xuntrans = 0.5*(X[:,:]+1)*(self.Xmax - self.Xmin) + self.Xmin
        return Xuntrans

class scaler_meanvar(object):
    """ 
    Scale the data to have a mean of 0 and variance of 1.
    """
    def __init__(self):
        self.fitted = False

    def fit(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        self.Xmean = np.mean(X,axis=0)
        self.Xstd  = np.std(X,axis=0)
        self.fitted = True

    def transform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted: self.fit(X)
        eps = np.finfo(np.float64).tiny
        Xtrans = (X[:,:]-self.Xmean)/(self.Xstd+eps)
        return Xtrans 

    def untransform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted:
            raise Exception('scaler has not been fitted')
        eps = np.finfo(np.float64).tiny
        Xuntrans = X[:,:]*(self.Xstd+eps) + self.Xmean
        return Xuntrans

class scaler_custom(object):
    '''
    Scale the data with custom center and range
    '''
    def __init__(self, centers, ranges):
        self.centers = centers
        self.ranges = ranges
        self.fitted = True

    def transform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xtrans = (X - self.centers)/(self.ranges + eps)
        return Xtrans

    def untransform(self,X):
        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xuntrans = X * (self.ranges + eps) + self.centers
        return Xuntrans
