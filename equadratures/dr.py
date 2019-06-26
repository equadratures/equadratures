"""The functionalities of this script includes:
    1. A polynomial-based active subspaces recipe;
    2. An ordinary least squares linear technique;
    3. The variable projection subroutine.

References:
    - Honkason, J. and Constantine, P. (2018). Data-driven polynomial ridge approximation using variable projection. [online] Arxiv.org. `Paper <https://arxiv.org/abs/1702.05859>`_.
    - Constantine, P. G. (2015). Active subspaces: Emerging ideas for dimension reduction in parameter studies. SIAM.
"""
import numpy as np
import scipy
from .parameter import Parameter
from .poly import Poly
import scipy.io
from .basis import Basis
from scipy.linalg import orth, sqrtm
from time import time

class dr(object):
    def __init__(self, poly=None, training_input=None, training_output=None):
        if poly is not None:
            self.poly = poly
        if training_input is not None:
            self.training_input = training_input
        if training_output is not None:
            self.training_output = training_output

    def computeActiveSubspaces(self, poly=None, samples=None, alpha=None, k=None, bootstrap=False, bs_trials = 50):
        """
        Computes the eigenvalues and corresponding eigenvectors for the high dimensional kernel matrix via polynomial model.

        :param PolynomialObject: an instance of PolynomialObject class
        :param samples: ndarray, sample vector points, default to None
        :return:
            * **eigs (numpy array)**: Eigenvalues obtained
            * **eigVecs (numpy array)**: Eigenvectors obtained
        """
        if not(hasattr(self,'poly')) and poly is None:
            raise(Exception('Must declare poly!'))
        elif poly is None:
            poly = self.poly

        if samples is None:
            d = poly.dimensions
            if alpha is None:
                alpha = 4
            if k is None or k > d:
                k = d
            M = int(alpha * k * np.log(d))
            X = np.zeros((M, d))
            for j in range(0, d):
                X[:, j] =  np.reshape(poly.parameters[j].getSamples(M), M)
        else:
            X = samples
            M, d = X.shape

        # Gradient matrix!
        polygrad = poly.evaluatePolyGradFit(X)
        weights = np.ones((M, 1)) / M
        R = polygrad.transpose() * weights
        C = np.dot(polygrad, R )

        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]
        if bootstrap:
            all_bs_eigs = np.zeros((bs_trials, d))
            # all_bs_W = np.zeros((bs_trials, d, d))
            for t in range(bs_trials):
                bs_samples = X[np.random.randint(0, M, size=M), :]
                polygrad_bs = poly.evaluatePolyGradFit(bs_samples)
                weights_bs = np.ones((M, 1)) / M
                R_bs = polygrad_bs.transpose() * weights_bs
                C_bs = np.dot(polygrad_bs, R_bs)

                e_bs, W_bs = np.linalg.eigh(C_bs)
                all_bs_eigs[t,:] = np.flipud(e_bs)
                # eigVecs_bs = np.fliplr(W_bs)

            eigs_bs_lower = np.min(all_bs_eigs, axis = 0)
            eigs_bs_upper = np.max(all_bs_eigs, axis = 0)
            return eigs,eigVecs,eigs_bs_lower,eigs_bs_upper
        else:
            return eigs,eigVecs

    def linearModel(self, training_input=None, training_output=None, r=False):
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
        x,residual=np.linalg.lstsq(A,training_output)[:2]
        u=x[:D]
        c=x[D]
        if r:
            r2 = 1-residual/(training_output.size * training_output.var())
            return u,c,r2
        return u,c

    def standard(self,bounds,X=None):
        """
        Standardize the inputs before Dimension Reduction

        :param X: ndarray, the undimensioned input values
        :param bounds: ndarray, M-by-2 array, where the first column contains the lower bounds of
        the M variables, the second column the upper bounds.
        :return:
            * **X_stnd (numpy array)**: Standardized input values
        """
        update_self = False
        if X is None:
            X = self.training_input
            update_self = True
        M,m=X.shape
        X_stnd=np.zeros((M,m))
        for i in range(0,M):
            for j in range(0,m):
                X_stnd[i,j]=2*(X[i,j]-bounds[j,0])/(bounds[j,1]-bounds[j,0])-1
        if update_self:
            self.training_input = X_stnd.copy()
        return X_stnd


    def variable_projection(self,n,p,X=None,f=None,gamma=None,beta=None,tol=None,maxiter=1000,U0=None):
        """
        Variable Projection function to obtain an active subspace in inputs design space

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

        if gamma is None:
            gamma=0.1
        if beta is None:
            beta=1e-4
        if X is None:
            X = self.training_input
        if f is None:
            f = self.training_output
        if X is None or f is None:
            raise Exception('Missing training data!')
        M,m=X.shape
        if U0 is None:
            Z=np.random.randn(m,n)
            U,_=np.linalg.qr(Z)
        else:
            U = orth(U0)
        if tol is None:
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

        # print eta[:3]

        #Construct the Vandermonde matrix step 6
        V,Polybasis=vandermonde(eta,p)
        # print V[:3]
        V_plus=np.linalg.pinv(V)
        coeff=np.dot(V_plus,f)
        res=f-np.dot(V,coeff)
        R=np.linalg.norm(res)
        #TODO: convergence criterion??

        for iteration in range(0,maxiter):
            time_index = 0
            #Construct the Jacobian step 9
            J=jacobian_vp(V,V_plus,U,y,f,Polybasis,eta,minmax,X)
            # print J[0]
            #Calculate the gradient of Jacobian #step 10
            G=np.zeros((m,n))
            for i in range(0,M):
                G=G+res[i]*J[i,:,:]


            #conduct the SVD for J_vec
            # vec_J=np.zeros((M,(m*n)))
            # for i in range(0,M):
            #     for j in range(0,m):
            #         for k in range(0,n):
            #             vec_J[i,j*n+k]=J[i,j,k]
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


                V_new,Polybasis=vandermonde(eta,p)
                V_plus_new=np.linalg.pinv(V_new)
                coeff_new=np.dot(V_plus_new,f)
                res_new=f-np.dot(V_new,coeff_new)
                R_new=np.linalg.norm(res_new)

                if np.linalg.norm(res_new)<=np.linalg.norm(res)+alpha*beta*t or t<1e-10:#step 21
                    break
                t=t*gamma

            dist_change = subspace_dist(U, U_new)
            # print dist_change
            U = U_new
            V = V_new
            coeff = coeff_new
            V_plus = V_plus_new
            res = res_new
            R = R_new
            if not(tol is None):
                if dist_change < tol:
                    print("VP finished with %d iterations" % iteration)
                    break
        return U,R

    def vector_AS(self, list_of_polys, R = None, alpha=None, k=None, samples=None, bootstrap=False, bs_trials = 50
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
    Object=Basis('Total order',listing)
    #Establish n Parameter objects
    params=[]
    P=Parameter(order=p,lower=-1,upper=1,distribution='uniform')
    for i in range(0,n):
        params.append(P)
    #Use the params list to establish the Poly object
    Polybasis=Poly(params,Object)
    V=Polybasis.getPolynomial(eta)
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
    time_index = 0
    ti = time()


    M,N=V.shape
    m,n=U.shape
    Gradient=Polybasis.getPolynomialGradient(eta)
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
        J[p,:,:] = list_of_poly[p].evaluatePolyGradFit(X)
    return J

def subspace_dist(U, V):
    if len(U.shape) == 1:
        return np.linalg.norm(np.outer(U, U) - np.outer(V, V), ord=2)
    else:
        return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)
