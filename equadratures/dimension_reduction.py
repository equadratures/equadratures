from equadratures.parameter import Parameter
from equadratures.basis import Basis
import numpy as np
import scipy
import scipy.io
from scipy.linalg import orth, sqrtm
from time import time

class Subspaces(object):
    """
    This class contains all the dimension reduction capabilities.

    :param Poly poly:
        An instance of the Poly class.

    """
    def __init__(self, poly, method, args=None):
        self.poly = poly
        self.method = method
        self.subspace_dimension = 2
        self.polynomial_degree = 2
        self.bootstrap_trials = None
        self.active_subspace_samples = None
        self.variable_projection_iterations = 1000
        self.args = args
        if self.args is not None:
            if 'subspace-dimension' in args: self.subspace_dimension = args.get('subspace-dimension')
            if 'bootstrap-replicates' in args: self.bootstrap_trials = args.get('bootstrap-replicates')
            if 'active-subspace-samples' in args: self.active_subspace_samples = args.get('active-subspace-samples')
            if 'variable-projection-iterations' in args: self.variable_projection_iterations = args.get('variable-projection-iterations')
        if self.method == 'active-subspaces':
            self.__get_active_subspaces()
        elif self.method == 'variable-projection':
            self.__get_variable_projection()
        elif self.method == 'linear-subspaces':
            self.__get_linear_subspaces()
        else:
            raise(ValueError)
    def get_eigenvalues(self):
        if self.method == 'active-subspaces':
            if self.bootstrap_trials is None:
                return self.__eigenvalues
            else:
                return self.__eigenvalues, self.__eigenvalues_bootstrap_lower, self.__eigenvalues_bootstrap_upper
        else:
            return None
    def get_reduced_subspace(self):
        return self.__active_subspace
    def get_remaining_subspace(self):
        return self.__inactive_subspace
    def __get_active_subspaces(self):
        """
        Computes the active subspace.

        """
        if self.poly.quadrature_points is None:
            d = poly.dimensions
            if alpha is None:
                alpha = 4
            if k is None or k > d:
                k = d
            M = int(alpha * k * np.log(d))
            X = np.zeros((M, d))
            for j in range(0, d):
                X[:, j] =  np.reshape(self.poly.parameters[j].get_samples(M), M)
        else:
            X = self.poly.quadrature_points
            M, d = X.shape
        polygrad = self.poly.get_polyfit_grad(X)
        weights = np.ones((M, 1)) / M
        R = polygrad.transpose() * weights
        C = np.dot(polygrad, R )
        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]
        if self.bootstrap_trials is not None:
            all_bs_eigs = np.zeros((self.bootstrap_trials, d))
            for t in range(self.bootstrap_trials):
                bs_samples = X[np.random.randint(0, M, size=M), :]
                polygrad_bs = poly.get_polyfit_grad(bs_samples)
                weights_bs = np.ones((M, 1)) / M
                R_bs = polygrad_bs.transpose() * weights_bs
                C_bs = np.dot(polygrad_bs, R_bs)
                e_bs, W_bs = np.linalg.eigh(C_bs)
                all_bs_eigs[t,:] = np.flipud(e_bs)
            eigs_bs_lower = np.min(all_bs_eigs, axis = 0)
            eigs_bs_upper = np.max(all_bs_eigs, axis = 0)
            self.__eigenvalues_bootstrap_lower = eigs_bs_lower
            self.__eigenvalues_bootstrap_upper = eigs_bs_upper
        self.__active_subspace = eigVecs[:, 0:self.subspace_dimension]
        self.__inactive_subspace = eigVecs[:, self.subspace_dimension::]
        self.__eigenvalues = eigs
    def __get_linear_subspaces(self):
        """
        Computes the coefficients for a linear model between inputs and outputs
        :param Xtrain: ndarray, the input values
        :param ytrain: array, the output values
        :return:
            * **u (numpy array)**: Coefficients correspond to each dimension
            * **c (double)**: Constant bias
        """
        if training_input is None:
            training_input = self.training_input
        if training_output is None:
            training_output = self.training_output
        if training_input is None or training_output is None:
            raise Exception('training data missing!')
        N,D=training_input.shape
        A=np.concatenate((training_input,np.ones((N,1))),axis=1)
        x,residual=np.linalg.lstsq(A,training_output, rcond=None)[:2]
        u=x[:D]
        c=x[D]
        if r:
            r2 = 1-residual/(training_output.size * training_output.var())
            return u,c,r2
        return u,c
    def __get_variable_projection(self, verbose=False):
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
        X = self.poly.quadrature_points
        f = self.poly.outputs
        n = self.subspace_dimension
        maxiter = self.variable_projection_iterations
        gamma=0.1
        beta=1e-4
        M,m=X.shape
        Z=np.random.randn(m,n)
        U,_=np.linalg.qr(Z)
        tol = 1e-7
        y=np.dot(X,U)
        minmax=np.zeros((2,n))
        for i in range(0,n):
            minmax[0,i]=min(y[:,i])
            minmax[1,i]=max(y[:,i])
        #Construct the affine transformation
        eta=np.zeros((M,n))
        for i in range(0,M):
            for j in range(0,n):
                eta[i,j]=2*(y[i,j]-minmax[0,j])/(minmax[1,j]-minmax[0,j])-1
        #Construct the Vandermonde matrix step 6
        # --> V,Polybasis=vandermonde(eta,self.polynomial_degree)

        V_plus=np.linalg.pinv(V)
        coeff=np.dot(V_plus,f)
        res=f-np.dot(V,coeff)
        R=np.linalg.norm(res)
        #TODO: convergence criterion??
        for iteration in range(0,maxiter):
            #Construct the Jacobian step 9
            J=jacobian_vp(V,V_plus,U,y,f,Polybasis,eta,minmax,X)
            #Calculate the gradient of Jacobian #step 10
            G=np.zeros((m,n))
            for i in range(0,M):
                G=G+res[i]*J[i,:,:]
            #conduct the SVD for J_vec
            vec_J = np.reshape(J,(M,m*n))
            Y,S,Z=np.linalg.svd(vec_J,full_matrices=False)#step 11
            #obtain delta
            delta = np.dot(Y[:,:-n**2].T, res)
            delta = np.dot(np.diag(1/S[:-n**2]), delta)
            delta = -np.dot(Z[:-n**2,:].T, delta).reshape(U.shape)
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
            for iter2 in range(0,50):
                U_new=np.dot(UZ, np.diag(np.cos(S*t))) + np.dot(Y, np.diag(np.sin(S*t)))#step 19

                U_new=orth(U_new)
                #Update the values with the new U matrix
                y=np.dot(X,U_new)
                minmax=np.zeros((2,n))
                for i in range(0,n):
                    minmax[0,i]=min(y[:,i])
                    minmax[1,i]=max(y[:,i])
                eta=np.zeros((M,n))
                for i in range(0,M):
                    for j in range(0,n):
                        eta[i,j]=2*(y[i,j]-minmax[0,j])/(minmax[1,j]-minmax[0,j])-1

                V_new,Polybasis=vandermonde(eta,self.polynomial_degree)
                V_plus_new=np.linalg.pinv(V_new)
                coeff_new=np.dot(V_plus_new,f)
                res_new=f-np.dot(V_new,coeff_new)
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
            if not(tol is None):
                if dist_change < tol:
                    if verbose:
                        print("VP finished with %d iterations" % iteration)
                    break
        if iteration == maxiter-1 and verbose:
            print("VP finished with %d iterations" % iteration)

        return U,R
def vandermonde(eta,p):
    """
    Internal function to variable_projection
    Calculates the Vandermonde matrix using polynomial basis functions
    :param eta: ndarray, the affine transformed projected values of inputs in active subspace
    :param p: int, the maximum degree of polynomials
    :return:
        * **V (numpy array)**: The resulting Vandermode matrix
        * **Polybasis (Poly object)**: An instance of Poly object containing the polynomial basis derived
    """
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
    Polybasis=Poly(params,Object)
    V=Polybasis.get_poly(eta)
    V=V.T
    return V,Polybasis
def jacobian_vp(V,V_plus,U,y,f,Polybasis,eta,minmax,X):
    """
    Internal function to variable_projection
    Calculates the Jacobian tensor using polynomial basis functions
    :param V: ndarray, the affine transformed outputs
    :param V_plus: ndarray, psuedoinverse matrix
    :param U: ndarray, the active subspace directions
    :param y: array, the untransformed projected values of inputs in active subspace
    :param f: array, the untransformed outputs
    :param Polybasis: Poly object, an instance of Poly class
    :param eta: ndarray, the affine transformed projected values of inputs in active subspace
    :param minmax: ndarray, the upper and lower bounds of input projections in each dimension
    :param X: ndarray, the input
    :return:
        * **J (ndarray)**: The Jacobian tensor
    """
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
    """
    Evaluates the Jacobian tensor for a list of polynomials.
    :param list_of_poly: list. Contains all m polynomials.
    :param X: ndarray, input points. N-by-d where N is the number of points, d the input dimension.
    :return:
        * **J (ndarray)**: The Jacobian tensor, m-by-d-by-N, where J[a,b,c] = dP_a/dx_c_b (b-th dimension of P_a's gradient at x_c)
    """
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