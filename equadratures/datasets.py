import os
import numpy as np
import scipy.stats as st
import requests
from io import BytesIO
from equadratures.scalers import scaler_minmax 

def gen_linear(n_observations=100, n_dim=5, n_relevent=5,bias=0.0, noise=0.0, random_seed=None):
    """Generate a synthetic linear dataset for regression. Data is generated using a random linear regression model with ``n_relevent`` input dimensions. Gaussian noise with standard deviation ``noise`` is added. 
    
    :param int n_observations: The number of observations (samples).
    :param int n_dim: The total number of dimensions.
    :param int n_relevent: The number of relevent input dimensions, i.e., the number of features used to build the linear model used to generate the output.
    :param float bias: The bias term in the underlying linear model.
    :param float noise: The standard deviation of the gaussian noise applied to the output.
    :param int random_seed: Random number generator seed. 
    
    :return:
    **X**: A numpy.ndarray of shape (n_observations,n_dim) containing the inputs.
    **y** : A numpy.ndarray of shape (n_observations,1) containing the outputs/targets.
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
    """Generates the friedman regression problem described by Friedman [1] and Breiman [2]. The function has n_dim=5, and choosing n_dim>5 adds irrelevent input dimensions. Gaussian noise with standard deviation ``noise`` is added. 
    
    :param int n_observations: The number of observations (samples).
    :param int n_dim: The total number of dimensions. n_dim>=5, with n_dim>5 adding irrelevent input dimensions.
    :param float noise: The standard deviation of the gaussian noise applied to the output.
    :param int random_seed: Random number generator seed. 
    :param boolean normalise: Normalise y to lie between -1 to 1. 
   
    :return:
    **X**: A numpy.ndarray of shape (n_observations,n_dim) containing the inputs.
    **y** : A numpy.ndarray of shape (n_observations,1) containing the outputs/targets.

    **References**
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

def load_eq_dataset(dataset,data_dir=None):
    # Check if valid dataset
    datasets = ['naca0012','blade_envelopes','probes']
    if dataset not in datasets:
        raise ValueError('dataset specified in load_eq_dataset not recognised, avaiable datasets: ', datasets)

    # Download from github
    if data_dir is None:
        print('Downloading the ' + dataset + ' dataset from github...') 
        # .npz file
        git_url = os.path.join('https://github.com/Effective-Quadratures/data-sets/raw/main/',dataset,dataset+'.npz')
        try:
            r = requests.get(git_url,stream=True)
            r.raise_for_status()
            data = np.load(BytesIO(r.raw.read()))
        except requests.exceptions.RequestException as e:  
            raise SystemExit(e)
        # .md file
        git_url = os.path.join('https://raw.githubusercontent.com/Effective-Quadratures/data-sets/main',dataset,'README.md')
        try:
            r = requests.get(git_url)
            r.raise_for_status()
            print('\n',r.text)
        except requests.exceptions.RequestException as e:  
            raise SystemExit(e)

    # If the user has cloned the data-sets repo and provided its location in data_dir
    else:
        print('Loading the dataset from ', data_dir)
        data = np.load(os.path.join(data_dir,dataset,dataset+'.npz'))
        f = open(os.path.join(data_dir,dataset,'README.md'))
        print(f.read())

    return data


def train_test_split(X,y,train=0.7,random_seed=None,shuffle=True):
    """Split arrays or matrices into random train and test subsets. Inspired by scikit-learn's datasets.train_test_split.
    :param numpy.ndarray X: An numpy.ndarray of shape (n_observations,n_dim) containing the inputs.
    :param numpy.ndarray y: A numpy.ndarray of shape (n_observations,1) containing the outputs/targets.
    :param float train: Fraction between 0.0 and 1.0, representing the proportion of the dataset to include in the train split. 
    :param boolean shuffle: Whether to shuffle the rows of data when spliting

    :return:
    **X_train**: A numpy.ndarray of shape (n_train,n_dim) containing the training inputs.
    **X_test**: A numpy.ndarray of shape (n_test,n_dim) containing the test inputs.
    **y_train**: A numpy.ndarray of shape (n_train,1) containing the training outputs/targets.
    **y_test**: A numpy.ndarray of shape (n_test,1) containing the test outputs/targets.
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

def score(y_true,y_pred,metric,X=None):
    """
    Evaluates the accuracy/error score between predictions ``y_pred`` and the truth ``y_true``, according to the given accuracy metric ``metric``. 

    :param numpy.ndarray y_true:
        An ndarray with shape (number_of_observations, 1), containing predictions.
    :param numpy.ndarray y_pred:
        An ndarray with shape (number_of_observations, 1) containing the true data.
    :param string metric:
        An optional string containing the scoring metric to use. Avaliable options are: ``adjusted_r2``, ``r2``, ``mae``, ``rmse``, or ``normalised_mae`` (default: ``adjusted_r2``). 

    :return:
        **score**: The accuracy or error score..
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
