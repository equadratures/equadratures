"""The polynomial parent class."""
from .stats import Statistics
from .parameter import Parameter
from .basis import Basis
from scipy.spatial import ConvexHull
from scipy.misc import comb
import numpy as np
from .poly import Poly

class Projectedpoly(Poly):
    """
    The class defines a Projectedpoly object.

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.

    """
    def __init__(self, parameters, basis, subspace):
        super(Polyreg, self).__init__(parameters, basis)
        self.subspace = subspace
        rows, cols = self.subspace.shape
        if rows > cols:
            self.reduced_dimensions = cols
        else:
            self.reduced_dimensions = rows
            self.subspace = self.subspace.T
        self.orders = []
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        if not self.basis.orders :
            self.basis.setOrders(self.orders)
        self.projOpt = None

def getNumOfVertices(W):
    """
    Function that returns the expected number of vertices of the zonotope.

    :param Projectedpoly self:
        An instance of the Projectedpoly class.
    :return:
        An integer N specifying the expected number of vertices of zonotope.
    Notes
    -----
    https://github.com/paulcon/active_subspaces/blob/master/active_subspaces/domains.py nzm

    """
    m, n = W.shape
    N = 0
    for i in range(n):
        N += comb(m-1,i)
    N = 2*N
    return int(N)
def getIntervalVertices(W):
    """
    Function that returns the endpoints of the zonotopes for a 1D subspace.

    :param Projectedpoly self:
        An instance of the Projectedpoly class.
    :return matrix Y:
        A matrix Y of the vertices of the zonotope in the reduced space.
    :return matrix X:
        A matrix X of the vertices of the zonotope in the full space.
    Notes
    -----
    https://github.com/paulcon/active_subspaces/blob/master/active_subspaces/domains.py interval_endpoints

    """
    m, n = W.shape
    assert n == 1
    y0 = np.dot(W.T, np.sign(W))[0]
    if y0 < -y0:
        yl, yu = y0, -y0
        xl, xu = np.sign(W), -np.sign(W)
    else:
        yl, yu = -y0, y0
        xl, xu = -np.sign(W), np.sign(W)
    Y = np.array([yl, yu]).reshape((2,1))
    X = np.vstack((xl.reshape((1,m)), xu.reshape((1,m))))
    return Y, X
def getZonotopeVertices(W, numSamples=10000, maxCount=100000):
    """
    Function that returns the vertices that describe the zonotope.

    :param Projectedpoly self:
        An instance of the Projectedpoly class.
    :param integer numSamples:
        An integer specifying the number of samples to take at each iteration.
    :param integer maxCount:
        An integer specifying the maximum number of iterations.
    :return matrix Y:
        A matrix Y of the vertices of the zonotope in the reduced space.
    :return matrix X:
        A matrix X of the vertices of the zonotope in the full space.
    Notes
    -----
    https://github.com/paulcon/active_subspaces/blob/master/active_subspaces/domains.py zonotope_vertices

    """
    m, n = W.shape
    totalVertices = getNumOfVertices(W)

    numSamples = int(numSamples)
    maxCount = int(maxCount)

    Z = np.random.normal(size=(numSamples, n))
    X = getUniqueRows(np.sign(np.dot(Z, W.transpose())))
    X = getUniqueRows(np.vstack((X, -X)))
    N = X.shape[0]

    count = 0
    while N < totalVertices:
        Z = np.random.normal(size=(numSamples, n))
        X0 = getUniqueRows(np.sign(np.dot(Z, W.transpose())))
        X0 = getUniqueRows(np.vstack((X0, -X0)))
        X = getUniqueRows(np.vstack((X, X0)))
        N = X.shape[0]
        count += 1
        if count > maxCount:
            break

    numVertices = X.shape[0]
    if totalVertices > numVertices:
        print('Warning: {} of {} vertices found.'.format(numVertices, totalVertices))

    Y = np.dot(X, W)
    return Y.reshape((numVertices, n)), X.reshape((numVertices, m))
def getZonotopeLinearInequalities(W):
    """
    Function that returns the linear inequalities that describe the zonotope.

    :param Projectedpoly self:
        An instance of the Projectedpoly class.
    :return matrix A:
        The matrix A defining the linear inequalities Ax <= b
    :return vector b:
        The vector b defining the linear inequalities Ax <= b

    """
    n = W.shape[1]
    if n == 1:
        Y, X = getIntervalVertices(W)
    else:
        Y, X = getZonotopeVertices(W)
    return getHull(Y,X)
def getHull(Y,X):
    """
    Function that returns a dictionary of objects from hull computations.

    :param matrix Y:
        A matrix Y of points in the reduced space.
    :param matrix X:
        A matrix X of point in the full space, corresponding to points in Y.
    :return:
        A dictionary 'hull' containing the linear inequalities Ax <= b for the
        convex hull of the vertices in the reduced space 'vertV', as well as the
        corresponding vertices in the full space 'vertX'

    """
    n = Y.shape[1]
    if n == 1:
        A = np.array([[1],[-1]])
        b = np.array([[max(Y)],[min(Y)]])
        vertX = np.array([[np.argmax(Y)],[np.argmin(Y)]])
        return {'A': A, 'b': b, 'vertV': b, 'vertX': vertX}
    else:
        convexHull = ConvexHull(Y)
        A = convexHull.equations[:,:n]
        b = -convexHull.equations[:,n]
        vertV = Y[convexHull.vertices,:]
        vertX = X[convexHull.vertices,:]
        return {'A': A, 'b': b, 'vertV': vertV, 'vertX': vertX}
def getUniqueRows(X0):
    """
    Function that returns unique rows from ndarray.

    :param matrix X0:
        A matrix which may have multiple equivalent rows
    :return:
        A matrix X1 containing only the unique rows of X0
    Notes
    -----
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

    """
    X1 = X0.view(np.dtype((np.void, X0.dtype.itemsize * X0.shape[1])))
    return np.unique(X1).view(X0.dtype).reshape(-1, X0.shape[1])
