""" 
Utilities for downloading or generating datasets, splitting data, computing accuracy metrics, and scaling data.
"""
import os
import numpy as np
import scipy.stats as st
import requests
import posixpath
from io import BytesIO

def gen_linear(n_observations=100, n_dim=5, n_relevent=5,bias=0.0, noise=0.0, random_seed=None):
    """ Generate a synthetic linear dataset for regression. 

    Data is generated using a random linear regression model with ``n_relevent`` input dimensions. 
    The remaining dimensions are "irrelevent" noise i.e. they do not affect the output.
    Gaussian noise with standard deviation ``noise`` is added. 
  
    Parameters
    ----------
    n_observations : int, optional
        The number of observations (samples).
    n_dim : int, optional
        The total number of dimensions.
    n_relevent : int, optional 
        The number of relevent input dimensions, i.e., the number of features used to build the linear model used to generate the output.
    bias : float, optional 
        The bias term in the underlying linear model.
    noise : float, optional 
        The standard deviation of the gaussian noise applied to the output.
    random_seed : int, optional 
        Random number generator seed. 
    
    Returns
    -------
    tuple
        Tuple (X,y) containing two numpy.ndarray's; One with shape (n_observations,n_dim) containing the inputs, 
        and one with shape (n_observations,1) containing the outputs/targets.
    """
    # Generate input data
    n_relevent = min(n_dim, n_relevent)
    if np.__version__ >= '1.17':
        generator = np.random.default_rng(random_seed)
    else:
        generator = np.random.RandomState(random_seed)
    X = generator.standard_normal((n_observations,n_dim))
    X = scaler_minmax().transform(X)

    # Generate the truth model with n_relevent input dimensions
    truth_model = np.zeros((n_dim, 1))
#    truth_model[:n_relevent, :] = generator.standard_normal((n_relevent,1))
    truth_model[:n_relevent, :] = generator.uniform(-1,1,n_relevent).reshape(-1,1)
    y = scaler_minmax().transform(np.dot(X, truth_model)) + bias

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    return X, y

def gen_friedman(n_observations=100, n_dim=5, noise=0.0, random_seed=None,normalise=False):
    """ Generates the friedman regression problem described by Friedman [1] and Breiman [2]. 

    Inspired by :obj:`sklearn.datasets.make_friedman1`. The function has ``n_dim=5``, and choosing ``n_dim>5`` adds irrelevent input dimensions. 
    Gaussian noise with standard deviation ``noise`` is added. 
    
    Parameters
    ----------
    n_observations : int, optional 
        The number of observations (samples).
    n_dim : int, optional 
        The total number of dimensions. n_dim>=5, with n_dim>5 adding irrelevent input dimensions.
    noise : float, optional
        The standard deviation of the gaussian noise applied to the output.
    random_seed : int, optional
        Random number generator seed. 
    normalise : bool, optional 
        Normalise y to lie between -1 to 1. 
   
    Returns
    -------
    tuple
        Tuple (X,y) containing two numpy.ndarray's; One with shape (n_observations,n_dim) containing the inputs, 
        and one with shape (n_observations,1) containing the outputs/targets.

    References
    ----------
    1. J. Friedman, "Multivariate adaptive regression splines", The Annals of Statistics 19 (1), pages 1-67, 1991.
    2. L. Breiman, "Bagging predictors", Machine Learning 24, pages 123-140, 1996.
    """

    if n_dim < 5:
        raise ValueError("n_dim must be at least five.")

    if np.__version__ >= '1.17':
        generator = np.random.default_rng(random_seed)
    else:
        generator = np.random.RandomState(random_seed)
    X = generator.standard_normal((n_observations,n_dim))
    X = scaler_minmax().transform(X)

    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
        + 10 * X[:, 3] + 5 * X[:, 4] 
    y+= noise*np.std(y)*generator.standard_normal(n_observations)
    if normalise:
        y = scaler_minmax().transform(y.reshape(-1,1))
    
    return X,y

def load_eq_dataset(dataset,data_dir=None,verbose=True):
    """
    Loads the requested dataset from the `equadratures dataset repository <https://github.com/Effective-Quadratures/data-sets>`__. 

    Visit the aforementioned repo for a description of the available datasets.

    The requested dataset can either be downloaded directly upon request, or to minimise downloads the repo can be cloned 
    once by the user, and the local repo directory can be given via ``data_dir`` (see examples).

    Parameters
    ----------
    dataset : str
        The dataset to download. Options are ```naca0012```, ```blade_envelopes```, ```probes```, ```3Dfan_blades```, ```LS89_turbine```.
    data_dir : str, optional
        Directory name where a local clone of the data-sets repo is located. If given, the dataset will be loaded from here 
        instead of downloading from the remote repo.
    verbose: bool, optional
        Option to print verbose messages to screen.

    Returns
    -------
    NpzFile
        NpzFile instance (see `numpy.lib.format <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format>`__)
        containing the dataset. Contents can be accessed in the usual way e.g. ``X = NpzFile['X']``.

    Examples
    --------
    Loading from remote repository
        >>> # Load the naca0012 aerofoil dataset
        >>> data = eq.datasets.load_eq_dataset('naca0012')
        >>> print(data.files)
        ['X', 'Cp', 'Cl', 'Cd']
        >>> X = data['X']
        >>> y = data['Cp']

    Loading from a locally cloned repository
        >>> git clone https://github.com/Effective-Quadratures/data-sets.git
        >>> data = eq.datasets.load_eq_dataset('naca0012', data_dir='/Users/user/Documents/data-sets')
    """
    # Check if valid dataset
    datasets = ['naca0012','blade_envelopes','probes', '3Dfan_blades', 'LS89_turbine']
    if dataset not in datasets:
        raise ValueError('dataset specified in load_eq_dataset not recognised, avaiable datasets: ', datasets)

    # Download from github
    if data_dir is None:
        print('Downloading the ' + dataset + ' dataset from github...') 
        # .npz file
        git_url = posixpath.join('https://github.com/Effective-Quadratures/data-sets/raw/main/',dataset,dataset+'.npz')
        try:
            r = requests.get(git_url,stream=True)
            r.raise_for_status()
            data = np.load(BytesIO(r.raw.read()))
        except requests.exceptions.RequestException as e:  
            raise SystemExit(e)
        # .md file
        git_url = posixpath.join('https://raw.githubusercontent.com/Effective-Quadratures/data-sets/main',dataset,'README.md')
        try:
            r = requests.get(git_url)
            r.raise_for_status()
            if verbose: print('\n',r.text)
        except requests.exceptions.RequestException as e:  
            raise SystemExit(e)

    # If the user has cloned the data-sets repo and provided its location in data_dir
    else:
        print('Loading the dataset from ', data_dir)
        data = np.load(posixpath.join(data_dir,dataset,dataset+'.npz'))
        f = open(posixpath.join(data_dir,dataset,'README.md'))
        if verbose: print(f.read())

    return data


def train_test_split(X,y,train=0.7,random_seed=None,shuffle=True):
    """ Split arrays or matrices into random train and test subsets. 

    Inspired by :obj:`sklearn.model_selection.train_test_split`.

    Parameters
    ----------
    X : numpy.ndarray
        Array with shape (n_observations,n_dim) containing the inputs.
    y : numpy.ndarray 
        Array with shape (n_observations,1) containing the outputs/targets.
    train : float, optional
        Fraction between 0.0 and 1.0, representing the proportion of the dataset to include in the train split. 
    random_seed : int, optional
        Seed for random number generator.
    shuffle : bool, optional
        Whether to shuffle the rows of data when spliting.

    Returns
    -------
    tuple
        Tuple (X_train, X_test, y_train, y_test) containing the split data, output as numpy.ndarray's.

    Example
    -------
    >>> X_train, X_test, y_train, y_test = eq.datasets.train_test_split(X, y, 
    >>>                                    train=0.8, random_seed = 42)
    """
    if X.shape[0] == y.shape[0]:
        n_observations = X.shape[0]
    else:
        raise ValueError("X and y have different numbers of rows")
    if train > 0 or train < 1:
        n_train = int(train*n_observations)
        n_test  = n_observations - n_train    
    else:
        raise ValueError("train should be between 0 and 1")
    if shuffle:
        if np.__version__ >= '1.17':
            generator = np.random.default_rng(random_seed)
        else:
            generator = np.random.RandomState(random_seed)
        idx = generator.permutation(n_observations)
    else:
        idx = np.arange(n_observations)
    idx_train, idx_test = idx[:n_train], idx[n_train:]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]

def score(y_true,y_pred,metric='r2',X=None):
    """ Evaluates the accuracy/error score between predictions and the truth, according to the given accuracy metric.

    Parameters
    ----------
    y_pred : numpy.ndarray
        Array with shape (number_of_observations, 1), containing predictions.
    y_true : numpy.ndarray
        Array with shape (number_of_observations, 1) containing the true data.
    metric : str, optional
        The scoring metric to use. Avaliable options are: ```adjusted_r2```, ```r2```, ```mae```, ```rmse```, or ```normalised_mae```.
    X : numpy.ndarray
        The input data associated with **y_pred**. Required if ``metric=`adjusted_r2```.

    Returns
    -------
    float
        The accuracy or error score.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if metric == 'r2':
         score = st.linregress(y_true,y_pred)[2]**2
    elif metric == 'adjusted_r2':
        if X is None: raise ValueError('Must specify X in _score if adjusted_r2 metric used')
        N,d = X.shape
        r2 = st.linregress(y_true,y_pred)[2]**2
        score = 1.0 - (((1-r2)*(N-1))/(N-d-1))
    elif metric == 'mae':
        score = np.mean(np.abs(y_true-y_pred))
    elif metric == 'normalised_mae':
        score = np.mean(np.abs(y_true-y_pred))/np.std(y_true)
    elif metric == 'rmse':
        score = np.sqrt(np.mean((y_true-y_pred)**2))
    else:
        raise ValueError('Only r2, adjusted_r2, mae, normalised_mae, rmse scoring metrics currently supported')
    return score

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
    >>> X = eq.datasets.scaler_meanvar().transform(X)
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
    >>> scaler = eq.datasets.scaler_minmax()
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
