"""Solvers for computing the coefficients."""
import numpy as np

def least_squares(A, b):
    alpha = np.linalg.lstsq(A, b)
    return alpha[0]
def minimum_norm(A, b):
    return 0
def linear_system(A, b):
    return 0
def constrained_least_squares(A, b, C, d):
    return 0
def basis_pursuit_denoising(Ao, bo):
    A = deepcopy(Ao)
    b = deepcopy(bo)
    N = A.shape[0]
    # Possible noise levels
    log_epsilon = [-8,-7,-6,-5,-4,-3,-2,-1]
    epsilon = [float(10**i) for i in log_epsilon]
    errors = np.zeros(5)
    mean_errors = np.zeros(len(epsilon))
    # 5 fold cross validation
    for e in range(len(epsilon)):
        for n in range(5):
            indices = [int(i) for i in n * np.ceil(N/5.0) + range(int(np.ceil(N/5.0))) if i < N]
            A_ver = A[indices]
            A_train = np.delete(A, indices, 0)
            y_ver = y[indices].flatten()
            y_train = np.delete(y, indices).flatten()
            x_train = bp_denoise(A_train, y_train, epsilon[e])
            y_trained = np.reshape(np.dot(A_ver, x_train), len(y_ver))
            assert y_trained.shape == y_ver.shape
            errors[n] = np.mean(np.abs(y_trained - y_ver))/len(y_ver)
        mean_errors[e] = np.mean(errors)
    best_epsilon = epsilon[np.argmin(mean_errors)]
    x = bp_denoise(A, y, best_epsilon)
    residue = np.linalg.norm(np.dot(A, x).flatten() - y.flatten())
    return np.reshape(x, (len(x),1))
