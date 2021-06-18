"""
Classes to scale data. 

Some of these classes are called internally by other modules, but they can also be used independently as a pre-processing stage.

Scalers can fit to one set of data, and used to transform other data sets with the same number of dimensions.

Examples
--------
Fitting scaler implicitly during transform
    >>> # Define some 1D sample data
    >>> X = np.random.RandomState(0).normal(2,0.5,200)
    >>> (X.mean(),X.std())
    >>> (2.0354552465705806, 0.5107113843479977)
    >>>
    >>> # Scale to zero mean and unit variance
    >>> X = eq.scalers.scaler_meanvar().transform(X)
    >>> (X.mean(),X.std())
    >>> (2.886579864025407e-17, 1.0)

Using the same scaling to transform train and test data
    >>> # Define some 5D example data
    >>> X = np.random.RandomState(0).uniform(-10,10,size=(50,5))
    >>> y = X[:,0]**2 - X[:,4]
    >>> # Split into train/test
    >>> X_train, X_test,y_train,y_test = eq.datasets.train_test_split(X,y,train=0.7,random_seed=0)
    >>> (X_train.min(),X_train.max())
    >>> (-9.906090476149059, 9.767476761184525)
    >>>
    >>> # Define a scaler and fit to training split
    >>> scaler = eq.scalers.scaler_minmax()
    >>> scaler.fit(X_train)
    >>>
    >>> # Transform train and test data with same scaler
    >>> X_train = scaler.transform(X_train)
    >>> X_test = scaler.transform(X_test)
    >>> (X_train.min(),X_train.max())
    >>> (-1.0, 1.0)
    >>>
    >>> # Finally, e.g. of transforming data back again
    >>> X_train = scaler.untransform(X_train)
    >>> (X_train.min(),X_train.max())
    >>> (-9.906090476149059, 9.767476761184525)
"""
import numpy as np

class scaler_minmax(object):
    """ Scale the data to have a min/max of -1 to 1. """
    def __init__(self):
        self.fitted = False

    def fit(self,X):
        """ Fit scaler to data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to fit scaler to.
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        self.Xmin = np.min(X,axis=0)
        self.Xmax = np.max(X,axis=0)
        self.fitted = True

    def transform(self,X):
        """ Transforms data. Calls :meth:`~equadratures.scalers.scaler_minmax.fit` fit internally if scaler not already fitted.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to transform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing transformed data.
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted: self.fit(X)
        Xtrans = 2.0 * ( (X[:,:]-self.Xmin)/(self.Xmax - self.Xmin) ) - 1.0
        return Xtrans 

    def untransform(self,X):
        """ Untransforms data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to untransform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing untransformed data.

        Raises
        ------
        Exception 
            scaler has not been fitted
        """
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
        """ Fit scaler to data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to fit scaler to.
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        self.Xmean = np.mean(X,axis=0)
        self.Xstd  = np.std(X,axis=0)
        self.fitted = True

    def transform(self,X):
        """ Transforms data. Calls :meth:`~equadratures.scalers.scaler_meanvar.fit` fit internally if scaler not already fitted.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to transform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing transformed data.
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted: self.fit(X)
        eps = np.finfo(np.float64).tiny
        Xtrans = (X[:,:]-self.Xmean)/(self.Xstd+eps)
        return Xtrans 

    def untransform(self,X):
        """ Untransforms data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to untransform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing untransformed data.

        Raises
        ------
        Exception 
            scaler has not been fitted
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        if not self.fitted:
            raise Exception('scaler has not been fitted')
        eps = np.finfo(np.float64).tiny
        Xuntrans = X[:,:]*(self.Xstd+eps) + self.Xmean
        return Xuntrans

class scaler_custom(object):
    """ Scale the data by the provided offset and divisor.
    
    Parameters
    ----------
    offset : float, numpy.ndarray
        Offset to subtract from data. Either a float, or array with shape (number_of_samples, number_of_dimensions).
    div : float, numpy.ndarray
        Divisor to divide data with. Either a float, or array with shape (number_of_samples, number_of_dimensions).
    """
    def __init__(self, offset, div):
        self.offset = offset
        self.div = div
        self.fitted = True

    def transform(self,X):
        """ Transforms data. 

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to transform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing transformed data.
        """
        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xtrans = (X - self.offset)/(self.div + eps)
        return Xtrans

    def untransform(self,X):
        """ Untransforms data.

        Parameters
        ----------
        X : numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing data to untransform.

        Returns
        -------
        numpy.ndarray
            Array with shape (number_of_samples, number_of_dimensions) containing untransformed data.
        """

        if X.ndim == 1: X = X.reshape(-1,1)
        eps = np.finfo(np.float64).tiny
        Xuntrans = X * (self.div + eps) + self.offset
        return Xuntrans
