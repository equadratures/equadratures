"""The polynomial parent class."""
from .stats import Statistics
from .parameter import Parameter
from .basis import Basis
from .optimization import Optimization
from scipy.spatial import ConvexHull
from scipy.misc import comb
from scipy.spatial.distance import cdist
import numpy as np
VERSION_NUMBER = 7.6

class Projectedpoly(object):
    """
    The class defines a Projectedpoly object.

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.

    """
    def __init__(self, parameters, basis, subspace):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(parameters)
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
    def __setFunctionEvaluations__(self, function_evaluations):
        """
        Sets the function evaluations for the polynomial. This function can be called by the children of Projectedpoly.

        """
        self.function_evaluations = function_evaluations
    def __setCoefficients__(self, coefficients):
        """
        Sets the coefficients for polynomial. This function will be called by the children of Projectedpoly.

        :param Projectedpoly self:
            An instance of the Poly class.
        :param array coefficients:
            An array of the coefficients computed using either integration, least squares or compressive sensing routines.

        """
        self.coefficients = coefficients
    def __setBasis__(self, basisNew):
        """
        Sets the basis
        """
        self.basis = basisNew 
    def __setQuadrature__(self, quadraturePoints, quadratureWeights):
        """
        Sets the quadrature points and weights

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param matrix quadraturePoints:
            A numpy matrix filled with the quadrature points.
        :param matrix quadratureWeights:
            A numpy matrix filled with the quadrature weights.
        """
        self.quadraturePoints = quadraturePoints
        self.quadratureWeights = quadratureWeights
    def __setDesignMatrix__(self, designMatrix):
        """
        Sets the design matrix assocaited with the quadrature (depending on the technique) points and the polynomial basis.

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param matrix designMatrix:
            A numpy matrix filled with the multivariate polynomial evaluated at the quadrature points.

        """
        self.designMatrix = designMatrix
    def clone(self):
        """
        Clones a Projectedpoly object.

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :return:
            A clone of the Projectedpoly object.
        """
        return type(self)(self.parameters, self.basis)
    def approxFullSpacePolynomial(self):
        """
        Use the quadratic program to approximate the polynomial over the full space.
        """
        Polyfull = Poly()
        return Polyfull
    def getPolynomial(self, stackOfPoints, customBases=None):
        """
        Evaluates the value of each polynomial basis function at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the multivariate (in d-dimensions) polynomial basis functions must be evaluated.
        :return:
            A P-by-N matrix of polynomial basis function evaluations at the stackOfPoints, where P is the cardinality of the basis.
        """
        if customBases is None:
            basis = self.basis.elements
        else:
            basis = customBases
        basis_entries, dimensions = basis.shape

        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, __ = stackOfPoints.shape
        p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , _ , _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        polynomial = np.ones((basis_entries, no_of_points))
        for k in range(dimensions):
            basis_entries_this_dim = basis[:, k].astype(int)
            polynomial *= p[k][basis_entries_this_dim]

        return polynomial 
    def getPolynomialGradient(self, stackOfPoints, dim_index = None):
        """
        Evaluates the gradient for each of the polynomial basis functions at a set of points,
        with respect to each input variable.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _ , dpoly , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis) ) )
            return dpoly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , dp[i] , _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        R = []
        if dim_index is None:
            dim_index = range(dimensions)
        for v in range(dimensions):
            if not(v in dim_index):
                R.append(np.zeros((basis_entries, no_of_points)))
            else:
                polynomialgradient = np.ones((basis_entries, no_of_points))
                for k in range(dimensions):
                    basis_entries_this_dim = basis[:,k].astype(int)
                    if k==v:
                        polynomialgradient *= dp[k][basis_entries_this_dim]
                    else:
                        polynomialgradient *= p[k][basis_entries_this_dim]
                R.append(polynomialgradient)

        return R   
    def getPolynomialHessian(self, stackOfPoints):
        """
        Evaluates the hessian for each of the polynomial basis functions at a set of points,
        with respect to each input variable.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the gradient of the multivariate (in d-dimensions) polynomial basis
            functions must be evaluated.
        :return:
            A list with d^2 elements, each with a P-by-N matrix of polynomial evaluations at the stackOfPoints,
            where P is the cardinality of the basis.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}
        d2p = {}
                        
        # Save time by returning if univariate!
        if dimensions == 1:
            _ , _ , d2poly =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis) ) )
            return d2poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , dp[i] , d2p[i] = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i]) + 1 ) )    
        H = []
        for w in range(0, dimensions):
            gradDirection1 = w
            for v in range(0, dimensions):
                gradDirection2 = v
                polynomialhessian = np.zeros((basis_entries, no_of_points))
                for i in range(0, basis_entries):
                    temp = np.ones((1, no_of_points))
                    for k in range(0, dimensions):
                        if k == gradDirection1 == gradDirection2:
                            polynomialhessian[i,:] = d2p[k][int(basis[i,k])] * temp
                        elif k == gradDirection1:
                            polynomialhessian[i,:] = dp[k][int(basis[i,k])] * temp
                        elif k == gradDirection2:
                            polynomialhessian[i,:] = dp[k][int(basis[i,k])] * temp
                        else:
                            polynomialhessian[i,:] = p[k][int(basis[i,k])] * temp
                        temp = polynomialhessian[i,:]
                H.append(polynomialhessian)

        return H
    def getTensorQuadratureRule(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.

        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.basis.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._getLocalQuadrature(orders[u])
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        points = pp[:,1::]
        weights = ww

        # Return tensor grid quad-points and weights
        return points, weights
    def getQuadratureRule(self, options=None, number_of_points = None):
        """
        Generates quadrature points and weights.

        :param Poly self:
            An instance of the Poly class.
        :param string options:
            Two options exist for this string. The user can use 'qmc' for a distribution specific Monte Carlo (QMC) or they can use 'tensor grid' for standard tensor product grid. Typically, if the number of dimensions is less than 8, the tensor grid is the default option selected.
        :param int number_of_points:
            If QMC is chosen, specifies the number of quadrature points in each direction. Otherwise, this is ignored.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        if options is None:
            if self.dimensions > 5 or np.max(self.orders) > 4:
                options = 'qmc'
            else:
                options = 'tensor grid'
        if options.lower() == 'qmc':
            if number_of_points is None:
                default_number_of_points = 20000
            else:
                default_number_of_points = number_of_points
            p = np.zeros((default_number_of_points, self.dimensions))
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            for i in range(0, self.dimensions):
                p[:,i] = np.array(self.parameters[i].getSamples(default_number_of_points)).reshape((default_number_of_points,))
            return p, w

        if options.lower() == 'tensor grid' or options.lower() == 'quadrature':
            p,w = self.getTensorQuadratureRule([i for i in self.basis.orders])
            return p,w
    def evaluatePolyFit(self, stackOfPoints):
        """
        Evaluates the the polynomial approximation of a function (or model data) at prescribed points.

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A 1-by-N matrix of the polynomial approximation.
        """
        return self.getPolynomial(stackOfPoints).T *  np.mat(self.coefficients)
    def evaluatePolyGradFit(self, stackOfPoints, dim_index = None):
        """
        Evaluates the gradient of the polynomial approximation of a function (or model data) at prescribed points.

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A d-by-N matrix of the gradients of the polynomial approximation.

        **Notes:**

        This function should not be confused with getPolynomialGradient(). The latter is only concerned with approximating what the multivariate polynomials
        gradient values are at prescribed points.
        """
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        H = self.getPolynomialGradient(stackOfPoints, dim_index=dim_index)
        grads = np.zeros((self.dimensions, no_of_points ) )
        if self.dimensions == 1:
            return np.mat(self.coefficients).T * H
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads
    def evaluatePolyHessFit(self, stackOfPoints):
        """
        Evaluates the hessian of the polynomial approximation of a function (or model data) at prescribed points.

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A d-by-d-by-N matrix of the hessian of the polynomial approximation.
        """
        if stackOfPoints.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stackOfPoints.shape
        H = self.getPolynomialHessian(stackOfPoints)
        hess = np.zeros( (self.dimensions, self.dimensions,no_of_points) )
        for i in range(0, self.dimensions):
            for j in range(0, self.dimensions):
                hess[i,j,:] = np.mat(self.coefficients).T * H[i * self.dimensions + j]
        return hess
    def getPolyFitFunction(self):
        """
        Returns a callable polynomial approximation of a function (or model data).

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :return:
            A callable function.

        """
        return lambda x: np.array(self.getPolynomial(x).T *  np.mat(self.coefficients))
    def getPolyGradFitFunction(self):
        """
        Returns a callable for the gradients of the polynomial approximation of a function (or model data).

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :return:
            A callable function.

        """
        return lambda x : self.evaluatePolyGradFit(x)
    def getPolyHessFitFunction(self):
        """
        Returns a callable for the hessian of the polynomial approximation of a function (or model data).

        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :return:
            A callable function.

        """
        return lambda (x) : self.evaluatePolyHessFit(x)
    def getNumOfVertices(self):
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
        m, n = self.subspace.shape
        N = 0
        for i in range(n):
            N += comb(m-1,i)
        N = 2*N
        return int(N)
    def getIntervalVertices(self):
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
        W = self.subspace
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
    def getZonotopeVertices(self, numSamples=10000, maxCount=100000):
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
        W = self.subspace
        m, n = W.shape
        totalVertices = self.getNumOfVertices()
        
        numSamples = int(numSamples)
        maxCount = int(maxCount)
        
        Z = np.random.normal(size=(numSamples, n))
        X = self.getUniqueRows(np.sign(np.dot(Z, W.transpose())))
        X = self.getUniqueRows(np.vstack((X, -X)))
        N = X.shape[0]
        
        count = 0
        while N < totalVertices:
            Z = np.random.normal(size=(numSamples, n))
            X0 = self.getUniqueRows(np.sign(np.dot(Z, W.transpose())))
            X0 = self.getUniqueRows(np.vstack((X0, -X0)))
            X = self.getUniqueRows(np.vstack((X, X0)))
            N = X.shape[0]
            count += 1
            if count > maxCount:
                break
        
        numVertices = X.shape[0]
        if totalVertices > numVertices:
            print 'Warning: {} of {} vertices found.'.format(numVertices, totalVertices)
        
        Y = np.dot(X, W)
        return Y.reshape((numVertices, n)), X.reshape((numVertices, m))
    def getZonotopeLinearInequalities(self):
        """
        Function that returns the linear inequalities that describe the zonotope.
        
        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :return matrix A:
            The matrix A defining the linear inequalities Ax <= b
        :return vector b:
            The vector b defining the linear inequalities Ax <= b

        """
        n = self.subspace.shape[1]
        if n == 1:
            Y, X = self.getIntervalVertices()
        else:
            Y, X = self.getZonotopeVertices()
        return self.getHull(Y,X)
    def setProjection(self,bounds,W,dist=0.1):
        """
        Function that returns the hull of extreme points of projected set.
        
        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param vector bounds:
            A vector specifying the lower and upper bounds of polynomial.
        :param subspace W:
            A matrix W specifying the subspace we would like to project on to.
        :return dict P1:
            A dictionary object P1 containing the linear inequalities and the 
            vertices in reduced and full space of the projected set

        """
        self.defineInequalityCons(bounds)
        m, n = W.shape
        X = np.zeros((1,m))
        OK = 0
        MaxIter = 100
        cnt = 0
        
        while not OK:
            direction = np.random.uniform(-1,1,n)
            if cnt > MaxIter:
                raise Exception('Iterative hull algorithm exceeded maximum number of iterations.')
            x = self.maxDirectionOpt(direction,W)
            cnt += 1
            X = np.vstack((X,x))
            V = np.dot(X,W)
            V,ind = np.unique(V.round(decimals=3),return_index=True,axis=0)
            X = X[ind]
            if V.shape[0] == n+2:
                OK = 1
        X = X[~np.all(X == 0., axis=1)]
        V = V[~np.all(V == 0., axis=1)]
        
        P1 = self.getHull(V,X)
        OK = 0
        banDirections = []
        while not OK:
            for i in range(P1['A'].shape[0]):
                if P1['A'][i,:].tolist() not in banDirections:
                    if cnt > MaxIter:
                        print 'Exceeded number of maximum number of iterations'
                        return P1
                    direction = P1['A'][i,:]
                    x = self.maxDirectionOpt(direction,W)
                    cnt += 1
                    v = np.dot(x,W)
                    if min(cdist(v.reshape(1,-1),V)[0]) > dist:
                        X = np.vstack((X,x))
                        V = np.vstack((V,v))
                        V,ind = np.unique(V.round(decimals=3),return_index=True,axis=0)
                        X = X[ind]
                    else:
                        banDirections.append(P1['A'][i,:].tolist())
            P2 = self.getHull(V,X)
            if P1['vertV'].shape == P2['vertV'].shape:
                if np.allclose(P1['vertV'],P2['vertV']):
                    OK = 1
            P1 = P2
        return P1
    def defineInequalityCons(self,bounds):
        """
        Function that creates an optimization instance and adds the inequality constraints for
        the projection optimization problem.
        
        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param vector bounds:
            A vector specifying the lower and upper bounds of polynomial.

        """
        m = self.subspace.shape[0]
        opt = Optimization(method='trust-constr') 
        opt.addLinearIneqCon(np.eye(m),-np.ones(m),np.ones(m))
        opt.addNonLinearIneqCon({'poly':self,'bounds':bounds,'subspace':self.subspace})
        self.projOpt = opt
        return None
    def maxDirectionOpt(self,direction,U):
       """
        Function that creates an optimization instance and adds the inequality constraints for
        the projection optimization problem.
        
        :param Projectedpoly self:
            An instance of the Projectedpoly class.
        :param vector direction:
            A 1-by-n vector specifying the direction to maximise in.
        :param subspace U:
            A subspace matrix U in which to project onto.
        :return:
            A 1-by-m vector x which specifies the full dimensional answer to the
            maximum direction optimization problem.

        """
        n = U.shape[0]
        c = U.dot(direction)
        x0 = np.random.uniform(-1,1,n)
        objDict = {'function': lambda x: c.dot(x), 'jacFunction': lambda x: c, 'hessFunction': lambda x: np.zeros((n,n))}
        x = self.projOpt.optimizePoly(objDict,x0)['x']
        return x
    @staticmethod
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
    @staticmethod
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