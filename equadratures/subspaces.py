from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis
import numpy as np
import scipy
import scipy.io
from scipy.linalg import orth, sqrtm
from scipy.spatial import ConvexHull
from scipy.special import comb
from scipy.optimize import linprog
from time import time

class Subspaces(object):
    """
    This class defines a subspaces object. It can be used for polynomial-based subspace dimension reduction.

    :param string method: The method to be used for subspace-based dimension reduction. One option is ``active-subspace``, which uses
        ideas in [1] and [2] to compute a dimension-reducing subspace with a global polynomial approximant. Gradients evaluations of the polynomial
        approximation are used to compute the averaged outer product of the gradient covariance matrix. Another option is ``variable-projection`` [3],
        where a Gauss-Newton optimisation problem is solved to compute both the polynomial coefficients and its subspace.
    :param numpy.ndarray sample_points: A numpy ndarray with shape (number_of_observations, dimensions) that corresponds to a set of sample points over the parameter space.
    :param numpy.ndarray sample_outputs: A numpy ndarray with shape (number_of_observations, 1) that corresponds to model evaluations at the sample points.
    :param int polynomial_degree: The degree of the polynomial used in the subspace-based approximation.
    :param int subspace_dimension: The dimension of the *active* subspace.
    :param bool bootstrap: Bootstrap trials for computing the dimension reducing subspace.


    **Sample constructor initialisations**::

        import numpy as np
        from equadratures import *

        # Active subspaces with a global order 2 polynomial.
        mysubspace = Subspaces(method='active-subspace', sample_points=X_red, sample_outputs=Y_red)
        eigs = mysubspace.get_eigenvalues()
        W = mysubspace.get_subspace()
        e = mysubspace.get_eigenvalues()

    **References**
        1. Constantine, P., (2015) Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies. SIAM Spotlights.
        2. Seshadri, P., Shahpar, S., Constantine, P., Parks, G., Adams, M. (2018) Turbomachinery Active Subspace Performance Maps. Journal of Turbomachinery, 140(4), 041003. `Paper <http://turbomachinery.asmedigitalcollection.asme.org/article.aspx?articleid=2668256>`__.
        3. Hokanson, J., Constantine, P., (2018) Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM Journal of Scientific Computing, 40(3), A1566-A1589. `Paper <https://epubs.siam.org/doi/abs/10.1137/17M1117690>`__.
    """
    def __init__(self, method, full_space_poly=None, sample_points=None, sample_outputs=None, polynomial_degree=2, subspace_dimension=2, bootstrap=False, subspace_init=None, max_iter=1000, tol=None):
        self.full_space_poly = full_space_poly
        self.sample_points = sample_points
        self.Y = None # for the zonotope vertices
        if self.sample_points is not None:
            self.sample_points = standardise(sample_points)
        self.sample_outputs = sample_outputs
        self.method = method
        self.subspace_dimension = subspace_dimension
        self.polynomial_degree = polynomial_degree
        self.bootstrap = bootstrap
        if self.method.lower() == 'active-subspace' or self.method.lower() == 'active-subspaces':
            self.method == 'active-subspace'
            if self.full_space_poly is None:
                N, d = self.sample_points.shape
                param = Parameter(distribution='uniform', lower=-1, upper=1., order=self.polynomial_degree)
                myparameters = [param for _ in range(d)]
                mybasis = Basis("total-order")
                mypoly = Poly(myparameters, mybasis, method='least-squares', sampling_args={'sample-points':self.sample_points, \
                                                                    'sample-outputs':self.sample_outputs})
                mypoly.set_model()
                self.full_space_poly = mypoly
            self.sample_points = standardise(self.full_space_poly.get_points())
            self.sample_outputs = self.full_space_poly.get_model_evaluations()
            self._get_active_subspace()
        elif self.method == 'variable-projection':
            self._get_variable_projection(None,None,tol,max_iter,subspace_init,False)
    def get_subspace_polynomial(self):
        """
        Returns a polynomial defined over the dimension reducing subspace.

        :param Subspaces self:
            An instance of the Subspaces object.

        :return:
            **subspacepoly**: A Poly object that defines a polynomial over the subspace. The distribution of parameters is
            assumed to be uniform and the maximum and minimum bounds for each parameter are defined by the maximum and minimum values
            of the project samples.
        """
        active_subspace = self._subspace[:, 0:self.subspace_dimension]
        projected_points = np.dot(self.sample_points, active_subspace)
        myparameters = []
        for i in range(0, self.subspace_dimension):
            param = Parameter(distribution='uniform', lower=np.min(projected_points[:,i]), upper=np.max(projected_points[:,i]), \
                order=self.polynomial_degree)
            myparameters.append(param)
        mybasis = Basis("total-order")
        subspacepoly = Poly(myparameters, mybasis, method='least-squares', sampling_args={'sample-points':projected_points, \
                                                                    'sample-outputs':self.sample_outputs})
        subspacepoly.set_model()
        return subspacepoly
    def get_eigenvalues(self):
        """
        Returns the eigenvalues of the dimension reducing subspace. Note: this option is
        currently only valid for method ``active-subspace``.

        :param Subspaces self:
            An instance of the Subspaces object.

        :return:
            **eigenvalues**: A numpy.ndarray of shape (dimensions,) corresponding to the eigenvalues of the above mentioned covariance matrix.
        """
        if self.method == 'active-subspace':
            return self._eigenvalues
        else:
            print('Only the active-subspace method yields eigenvalues.')
    def get_subspace(self):
        """
        Returns the dimension reducing subspace.

        :param Subspaces self:
            An instance of the Subspaces object.

        :return:
            **subspace**: A numpy.ndarray of shape (dimensions, dimensions) where the first ``subspace_dimension`` columns
            contain the dimension reducing subspace, while the remaining columns contain its orthogonal complement.
        """
        return self._subspace
    def _get_active_subspace(self):
        """
        Active subspaces.
        """
        bs_trials=50
        if self.full_space_poly.get_points() is None:
            d = poly.dimensions
            if alpha is None:
                alpha = 4
            if k is None or k > d:
                k = d
            M = int(alpha * k * np.log(d))
            X = np.zeros((M, d))
            for j in range(0, d):
                X[:, j] =  np.reshape(self.full_space_poly.parameters[j].get_samples(M), M)
        else:
            X = self.full_space_poly.get_points()
            M, d = X.shape
        polygrad = self.full_space_poly.get_polyfit_grad(X)
        weights = np.ones((M, 1)) / M
        R = polygrad.transpose() * weights
        C = np.dot(polygrad, R )

        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]
        if self.bootstrap:
            all_bs_eigs = np.zeros((bs_trials, d))
            # all_bs_W = np.zeros((bs_trials, d, d))
            for t in range(bs_trials):
                bs_samples = X[np.random.randint(0, M, size=M), :]
                polygrad_bs = self.full_space_poly.get_polyfit_grad(bs_samples)
                weights_bs = np.ones((M, 1)) / M
                R_bs = polygrad_bs.transpose() * weights_bs
                C_bs = np.dot(polygrad_bs, R_bs)
                e_bs, W_bs = np.linalg.eigh(C_bs)
                all_bs_eigs[t,:] = np.flipud(e_bs)
            eigs_bs_lower = np.min(all_bs_eigs, axis = 0)
            eigs_bs_upper = np.max(all_bs_eigs, axis = 0)
            self._subspace = eigVecs
            self._eigenvalues = eigs
            self._eigenvalues_lower = eigs_bs_lower
            self._eigenvalues_upper = eigs_bs_upper
        else:
            self._subspace = eigVecs
            self._eigenvalues = eigs
    def _get_variable_projection(self, gamma,beta,tol,maxiter,U0, verbose):
        """
        Variable Projection function to obtain an active subspace in inputs design space
        Note: It may help to standardize outputs to zero mean and unit variance
        :param X: ndarray, the input
        :param f: array, the output
        :param n: int, number of active subspace directions to calculate
        :param p: int, degree of polynomials
        :param gamma: double, step length reduction factor (0,1)
        :param beta: double, Armijo tolerance for backtracking line search (0,1)
        :param tol: double, tolerance for convergence, measured in the norm of residual over norm of f
        :return:
            * **U (ndarray)**: The active subspace found
            * **R (double)**: Cost of deviation in fitting
        """
        # NOTE: How do we know these are the best values of gamma and beta?
        if gamma is None:
            gamma=0.1
        if beta is None:
            beta=1e-4
        M,m=self.sample_points.shape
        if U0 is None:
            Z=np.random.randn(m, self.subspace_dimension)
            U,_=np.linalg.qr(Z)
        else:
            U = orth(U0)
        if tol is None:
            tol = 1e-7
        y=np.dot(self.sample_points,U)
        minmax=np.zeros((2, self.subspace_dimension))
        minmax[0,:] = np.amin(y, axis=0)
        minmax[1,:] = np.amax(y, axis=0)
        #Construct the affine transformation
        eta = 2 * np.divide((y - minmax[0,:]), (minmax[1,:]-minmax[0,:])) - 1

        #Construct the _vandermonde matrix step 6
        V,Polybasis=vandermonde(eta, self.polynomial_degree)
        V_plus=np.linalg.pinv(V)
        coeff=np.dot(V_plus, self.sample_outputs)
        res= self.sample_outputs - np.dot(V,coeff)
        R=np.linalg.norm(res)
        #TODO: convergence criterion??

        for iteration in range(0,maxiter):
            #Construct the Jacobian step 9
            J=jacobian_vp(V,V_plus,U,y, self.sample_outputs,Polybasis,eta,minmax, self.sample_points)
            #Calculate the gradient of Jacobian #step 10
            G=np.zeros((m, self.subspace_dimension))
            # NOTE: Can be vectorised
            for i in range(0,M):
                G += res[i]*J[i,:,:]

            #conduct the SVD for J_vec
            vec_J = np.reshape(J,(M,m*self.subspace_dimension))
            Y,S,Z=np.linalg.svd(vec_J,full_matrices=False)#step 11

            #obtain delta
            delta = np.dot(Y[:,:-self.subspace_dimension**2].T, res)
            delta = np.dot(np.diag(1/S[:-self.subspace_dimension**2]), delta)
            delta = -np.dot(Z[:-self.subspace_dimension**2,:].T, delta).reshape(U.shape)

            #carry out Gauss-Newton step
            vec_delta=delta.flatten()# step 12

            #vectorize G step 13
            vec_G=G.flatten()
            alpha=np.dot(vec_G.T,vec_delta)
            norm_G=np.dot(vec_G.T,vec_G)

            #check alpha step 14
            if alpha>=0:
                delta=-G
                alpha=-norm_G

            #SVD on delta step 17
            
            Y,S,Z=np.linalg.svd(delta,full_matrices=False)

            UZ=np.dot(U,Z.T)
            t = 1
            for iter2 in range(0,20):
                U_new=np.dot(UZ, np.diag(np.cos(S*t))) + np.dot(Y, np.diag(np.sin(S*t)))#step 19
                U_new=orth(U_new)
                #Update the values with the new U matrix
                y=np.dot(self.sample_points, U_new)
                minmax[0,:] = np.amin(y, axis=0)
                minmax[1,:] = np.amax(y, axis=0)
                eta = 2 * np.divide((y - minmax[0,:]), (minmax[1,:]-minmax[0,:])) - 1

                V_new,Polybasis=vandermonde(eta, self.polynomial_degree)
                V_plus_new=np.linalg.pinv(V_new)
                coeff_new=np.dot(V_plus_new, self.sample_outputs)
                res_new= self.sample_outputs  -  np.dot(V_new,coeff_new)
                R_new=np.linalg.norm(res_new)

                if np.linalg.norm(res_new)<=np.linalg.norm(res)+alpha*beta*t or t<1e-10:#step 21
                    break
                t=t*gamma
            dist_change = subspace_dist(U, U_new)
            U = U_new
            V = V_new
            coeff = coeff_new
            V_plus = V_plus_new
            res = res_new
            R = R_new
            if dist_change < tol:
                if verbose:
                    print("VP finished with %d iterations" % iteration)
                break
        if iteration == maxiter-1 and verbose:
            print("VP finished with %d iterations" % iteration)
        active_subspace = U
        inactive_subspace = scipy.linalg.null_space(active_subspace.T)
        self._subspace = np.hstack([active_subspace, inactive_subspace])
    def get_zonotope_vertices(self, num_samples=10000, max_count=100000):
        """
        Returns the vertices of the zonotope -- the projection of the high-dimensional space over the computed
        subspace.

        :param Subspaces self:
            An instance of the Subspaces object.

        :return:
            **vertices**: A numpy.ndarray of shape (number of vertices, ``subspace_dimension``).

        **Note:**
        This routine has been adapted from Paul Constantine's zonotope_vertices() function; see reference below.

        Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L., (2016) Python Active-Subspaces Utility Library. Journal of Open Source Software, 1(5), 79. `Paper <http://joss.theoj.org/papers/10.21105/joss.00079>`__.
        """
        m = self._subspace.shape[0]
        n = self.subspace_dimension
        W = self._subspace[:, :n]
        if n == 1:
            y0 = np.dot(W.T, np.sign(W))[0]
            if y0 < -y0:
                yl, yu = y0, -y0
                xl, xu = np.sign(W), -np.sign(W)
            else:
                yl, yu = -y0, y0
                xl, xu = -np.sign(W), np.sign(W)
            Y = np.array([yl, yu]).reshape((2,1))
            X = np.vstack((xl.reshape((1,m)), xu.reshape((1,m))))
            self.Y = Y
            return Y
        else:
            total_vertices = 0
            for i in range(n):
                total_vertices += comb(m-1,i)
            total_vertices = int(2*total_vertices)

            Z = np.random.normal(size=(num_samples, n))
            X = get_unique_rows(np.sign(np.dot(Z, W.transpose())))
            X = get_unique_rows(np.vstack((X, -X)))
            N = X.shape[0]

            count = 0
            while N < total_vertices:
                Z = np.random.normal(size=(num_samples, n))
                X0 = get_unique_rows(np.sign(np.dot(Z, W.transpose())))
                X0 = get_unique_rows(np.vstack((X0, -X0)))
                X = get_unique_rows(np.vstack((X, X0)))
                N = X.shape[0]
                count += 1
                if count > max_count:
                    break

            num_vertices = X.shape[0]
            if total_vertices > num_vertices:
                print('Warning: {} of {} vertices found.'.format(num_vertices, total_vertices))

            Y = np.dot(X, W)
            self.Y = Y.reshape((num_vertices, n))
            return self.Y
    def get_linear_inequalities(self):
        """
        Returns the linear inequalities defining the zontope vertices, i.e., Ax<=b.

        :param Subspaces self:
            An instance of the Subspaces object.

        :return:
            **A**: The matrix for setting the linear inequalities.
        :return:
            **b**: The right-hand-side vector for setting the linear inequalities.
        """
        if self.Y is None:
            self.Y = self.get_zonotope_vertices()
        n = self.Y.shape[1]
        if n == 1:
            A = np.array([[1],[-1]])
            b = np.array([[max(self.Y)],[min(self.Y)]])
            return  A, b
        else:
            convexHull = ConvexHull(self.Y)
            A = convexHull.equations[:,:n]
            b = -convexHull.equations[:,n]
            return A, b
    def get_samples_constraining_active_coordinates(self, inactive_samples, active_coordinates):
        """

        A hit and run type sampling strategy for generating samples at a given coordinate in the active subspace
        by varying its coordinates along the inactive subspace.

        :param Subspaces self:
            An instance of the Subspaces object.
        :param int inactive_samples:
            The number of inactive samples required.
        :param numpy.ndarray active_coordiantes:
            The active subspace coordinates.

        :return:
            **X**: An numpy.ndarray of the full-space coordinates.

        **Note:**
        This routine has been adapted from Paul Constantine's hit_and_run() function; see reference below.

        Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L., (2016) Python Active-Subspaces Utility Library. Journal of Open Source Software, 1(5), 79. `Paper <http://joss.theoj.org/papers/10.21105/joss.00079>`__.

        """
        y = active_coordinates
        N = inactive_samples
        W1 = self._subspace[:, :self.subspace_dimension]
        W2 = self._subspace[:, self.subspace_dimension:]
        m, n = W1.shape

        s = np.dot(W1, y).reshape((m, 1))
        normW2 = np.sqrt(np.sum(np.power(W2, 2), axis=1)).reshape((m, 1))
        A = np.hstack((np.vstack((W2, -W2.copy())), np.vstack((normW2, normW2.copy()))))
        b = np.vstack((1 - s, 1 + s)).reshape((2 * m, 1))
        c = np.zeros((m - n + 1, 1))
        c[-1] = -1.0
        # print()

        zc = linear_program_ineq(c, -A, -b)
        z0 = zc[:-1].reshape((m - n, 1))

        # define the polytope A >= b
        s = np.dot(W1, y).reshape((m, 1))
        A = np.vstack((W2, -W2))
        b = np.vstack((-1 - s, -1 + s)).reshape((2 * m, 1))

        # tolerance
        ztol = 1e-6
        eps0 = ztol / 4.0

        Z = np.zeros((N, m - n))
        for i in range(N):

            # random direction
            bad_dir = True
            count, maxcount = 0, 50
            while bad_dir:
                d = np.random.normal(size=(m - n, 1))
                bad_dir = np.any(np.dot(A, z0 + eps0 * d) <= b)
                count += 1
                if count >= maxcount:
                    Z[i:, :] = np.tile(z0, (1, N - i)).transpose()
                    yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
                    return np.dot(self._subspace, yz).T

            # find constraints that impose lower and upper bounds on eps
            f, g = b - np.dot(A, z0), np.dot(A, d)

            # find an upper bound on the step
            min_ind = np.logical_and(g <= 0, f < -np.sqrt(np.finfo(np.float).eps))
            eps_max = np.amin(f[min_ind] / g[min_ind])

            # find a lower bound on the step
            max_ind = np.logical_and(g > 0, f < -np.sqrt(np.finfo(np.float).eps))
            eps_min = np.amax(f[max_ind] / g[max_ind])

            # randomly sample eps
            eps1 = np.random.uniform(eps_min, eps_max)

            # take a step along d
            z1 = z0 + eps1 * d
            Z[i, :] = z1.reshape((m - n,))

            # update temp var
            z0 = z1.copy()

        yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
        return np.dot(self._subspace, yz).T
def vector_AS(list_of_polys, R = None, alpha=None, k=None, samples=None, bootstrap=False, bs_trials = 50
                , J = None, save_path = None):
    # Find AS directions to vector val func
    # analogous to computeActiveSubspace
    # Since we are dealing with *one* vector val func we should have just one input space
    # Take the first of the polys.
    poly = list_of_polys[0]
    if samples is None:
        d = poly.dimensions
        if alpha is None:
            alpha = 4
        if k is None or k > d:
            k = d
        M = int(alpha * k * np.log(d))
        X = np.zeros((M, d))
        for j in range(0, d):
            X[:, j] = np.reshape(poly.parameters[j].getSamples(M), M)
    else:
        X = samples
        M, d = X.shape
    n = len(list_of_polys) # number of outputs
    if R is None:
        R = np.eye(n)
    elif len(R.shape) == 1:
        R = np.diag(R)
    if J is None:
        J = jacobian_vec(list_of_polys,X)
        if not(save_path is None):
            np.save(save_path,J)


    J_new = np.matmul(sqrtm(R), np.transpose(J,[2,0,1]))
    JtJ = np.matmul(np.transpose(J_new,[0,2,1]), J_new)
    H = np.mean(JtJ,axis=0)

    # Compute P_r by solving generalized eigenvalue problem...
    # Assume sigma = identity for now
    e, W = np.linalg.eigh(H)
    eigs = np.flipud(e)
    eigVecs = np.fliplr(W)
    if bootstrap:
        all_bs_eigs = np.zeros((bs_trials, d))
        all_bs_W = []
        for t in range(bs_trials):
            print("Starting bootstrap trial %d"%t)
            bs_samples = X[np.random.randint(0,M,size=M), :]
            J_bs = jacobian_vec(list_of_polys, bs_samples)
            J_new_bs = np.matmul(sqrtm(R), np.transpose(J_bs,[2,0,1]))
            JtJ_bs = np.matmul(np.transpose(J_new_bs, [0, 2, 1]), J_new_bs)
            H_bs = np.mean(JtJ_bs, axis=0)

            # Compute P_r by solving generalized eigenvalue problem...
            # Assume sigma = identity for now
            e_bs, W_bs = np.linalg.eigh(H_bs)
            all_bs_eigs[t,:] = np.flipud(e_bs)
            eigVecs_bs = np.fliplr(W_bs)
            all_bs_W.append(eigVecs_bs)

        eigs_bs_lower = np.min(all_bs_eigs, axis = 0)
        eigs_bs_upper = np.max(all_bs_eigs, axis = 0)
        return eigs,eigVecs,eigs_bs_lower,eigs_bs_upper, all_bs_W
    else:
        return eigs,eigVecs
def vandermonde(eta,p):
    _,n=eta.shape
    listing=[]
    for i in range(0,n):
        listing.append(p)
    Object=Basis('total-order',listing)
    #Establish n Parameter objects
    params=[]
    P=Parameter(order=p,lower=-1,upper=1,distribution='uniform')
    for i in range(0,n):
        params.append(P)
    #Use the params list to establish the Poly object
    Polybasis=Poly(params,Object, method='least-squares')
    V=Polybasis.get_poly(eta)
    V=V.T
    return V,Polybasis
def jacobian_vp(V,V_plus,U,y,f,Polybasis,eta,minmax,X):
    M,N=V.shape
    m,n=U.shape
    Gradient=Polybasis.get_poly_grad(eta)
    sub=(minmax[1,:]-minmax[0,:]).T# n*1 array
    vectord=np.reshape(2.0/sub,(n,1))
    #Initialize the tensor
    J=np.zeros((M,m,n))
    #Obtain the derivative of this tensor
    dV=np.zeros((m,n,M,N))
    for l in range(0,n):
        for j in range(0,N):
            current=Gradient[l].T
            if n==1:
                current=Gradient.T
            dV[:,l,:,j]=np.asscalar(vectord[l])*(X.T*current[:,j])#Eqn 16 17

    #Get the P matrix
    P=np.identity(M)-np.matmul(V,V_plus)
    V_minus=scipy.linalg.pinv(V)

    #Calculate entries for the tensor
    for j in range(0,m):
        for k in range(0,n):
            temp1=np.linalg.multi_dot([P,dV[j,k,:,:],V_minus])
            J[:,j,k]=(-np.matmul((temp1+temp1.T),f)).reshape((M,))# Eqn 15

    return J
def jacobian_vec(list_of_poly, X):
    m = len(list_of_poly)
    [N,d] = X.shape
    J = np.zeros((m,d,N))
    for p in range(len(list_of_poly)):
        J[p,:,:] = list_of_poly[p].get_polyfit_grad(X)
    return J
def subspace_dist(U, V):
    if len(U.shape) == 1:
        return np.linalg.norm(np.outer(U, U) - np.outer(V, V), ord=2)
    else:
        return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)
def standardise(X):
    M,d=X.shape
    X_stnd=np.zeros((M,d))
    for j in range(0,d):
        max_value = np.max(X[:,j])
        min_value = np.min(X[:,j])
        for i in range(0,M):
            X_stnd[i,j]=2.0 * ( (X[i,j]-min_value)/(max_value - min_value) ) -1
    return X_stnd
def unstandardise(X,X_orig):
    d=X.shape[1]
    X_unstnd=np.zeros_like(X)
    for j in range(0,d):
        max_value = np.max(X_orig[:,j])
        min_value = np.min(X_orig[:,j])
        X_unstnd[:,j] = 0.5*(X[:,j] +1)*(max_value - min_value) + min_value
    return X_unstnd
def linear_program_ineq(c, A, b):
    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make unbounded bounds
    bounds = []
    for i in range(c.size):
        bounds.append((None, None))

    A_ub, b_ub = -A, -b
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options={"disp": False}, method='simplex')
    if res.success:
        return res.x.reshape((c.size, 1))
    else:
        np.savez('bad_scipy_lp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                 c=c, A=A, b=b, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')
def get_unique_rows(X0):
    X1 = X0.view(np.dtype((np.void, X0.dtype.itemsize * X0.shape[1])))
    return np.unique(X1).view(X0.dtype).reshape(-1, X0.shape[1])
