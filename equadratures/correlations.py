""" Utilities for dealing with correlated inputs."""
from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis
import numpy as np
from scipy import stats
from scipy import optimize

class Correlations(object):
    """
    The class defines a Nataf transformation. The input correlated marginals are mapped from their physical space to a new
    standard normal space, in which points are uncorrelated.

    :param list D: List of parameters (distributions), interpreted here as the marginals.
    :param numpy.ndarray R: The correlation matrix associated with the joint distribution.

    **References**
        1. Melchers, R. E., (1945) Structural Reliability Analysis and Predictions. John Wiley and Sons, second edition.

    """
    def __init__(self, D=None, R=None):
        if D is None:
            raise(ValueError, 'Distributions must be given')
        else:
            self.D = D
        if R is None:
            raise(ValueError, 'Correlation matrix must be specified')
        else:
            self.R = R
        self.std = Parameter(order=5, distribution='normal',shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        inf_lim = -8.0
        sup_lim = - inf_lim
        p1 = Parameter(distribution = 'uniform', lower = inf_lim, upper = sup_lim, order = 31)
        myBasis = Basis('tensor-grid')
        Pols = Poly([p1, p1], myBasis, method='numerical-integration')
        p = Pols.get_points()
        w = Pols.get_weights() * (sup_lim - inf_lim)**2

        p1 = p[:,0]
        p2 = p[:,1]

        R0 = np.eye((len(self.D)))
        for i in range(len(self.D)):
            for j in range(i+1, len(self.D), 1):
                if self.R[i,j] == 0:
                    R0[i,j] = 0.0
                else:
                  tp11 = -(np.array(self.D[i].get_icdf(self.std.get_cdf(points=p1))) - self.D[i].mean ) / np.sqrt( self.D[i].variance )
                  tp22 = -(np.array(self.D[j].get_icdf(self.std.get_cdf(points=p2))) -  self.D[j].mean)/np.sqrt( self.D[j].variance )

                  rho_ij = self.R[i,j]
                  bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                  coefficientsIntegral = np.flipud(tp11*tp22 * w)

                  def check_difference(rho_ij):
                      bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                      diff = np.dot(coefficientsIntegral, bivariateNormalPDF)
                      return diff - self.R[i,j]

                  if (self.D[i].name!='custom') or (self.D[j].name!='custom'):
                    rho = optimize.newton(check_difference, self.R[i,j], maxiter=50)
                  else:
                    res = optimize.least_squares(check_difference, R[i,j], bounds=(-0.999,0.999), ftol=1.e-03)
                    rho = res.x
                    print('A Custom Marginal is present')

                  R0[i,j] = rho
                  R0[j,i] = R0[i,j]

        self.A = np.linalg.cholesky(R0)
        print('The Cholesky decomposition of fictive matrix R0 is:')
        print(self.A)
        print('The fictive matrix is:')
        print(R0)
    def get_uncorrelated_from_correlated(self, X):
        """
        Method for mapping correlated variables to a new standard space.

        :param Correlations self: An instance of the Correlations object.
        :param numpy.ndarray X: A numpy ndarray of shape (N,M) where input marginals are organized along columns; M represents the number of correlated marginals

        :return:
            **xu**: A numpy.ndarray of shape (N, M), which contains standardized uncorrelated data. The transformation of each i-th input marginal is stored along the i-th column of the output matrix.
        """
        c = X[:,0]
        w1 = np.zeros((len(c),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(c)):
                w1[j,i] = self.D[i].get_cdf(points=X[j,i])
                if (w1[j,i] >= 1.0):
                    w1[j,i] = 1.0 - 10**(-10)
                elif (w1[j,i] <= 0.0):
                    w1[j,i] = 0.0 + 10**(-10)
        sU = np.zeros((len(c),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(c)):
                sU[j,i] = self.std.get_icdf(w1[j,i])
        sU = np.array(sU)
        sU = sU.T
        xu = np.linalg.solve(self.A,sU)
        xu = np.array(xu)
        xu = xu.T
        return xu
    def get_correlated_from_uncorrelated(self, X):
        """
        Method for mapping uncorrelated variables from standard normal space to a new physical space in which variables are correlated.

        :param Correlations self: An instance of the Correlations object.
        :param numpy.ndarray X: Samples of uncorrelated points from the marginals; of shape (N,M)

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the correlated samples.
        """
        X = X.T

        invA = np.linalg.inv(self.A)
        Z = np.linalg.solve(invA, X)
        Z = Z.T

        xc = np.zeros((len(Z[:,0]), len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(Z[:,0])):
                xc[j,i] = self.std.get_cdf(points=Z[j,i])
        Xc = np.zeros((len(Z[:,0]),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(Z[:,0])):
                temporary = np.matrix(xc[j,i])
                temp = self.D[i].get_icdf(temporary)

                t = temp[0]
                Xc[j,i] = t
        return Xc
    def get_uncorrelated_samples(self, N=None):
        """
        Method for generating uncorrelated samples.

        :param int N: Number of uncorrelated samples required.
        :param numpy.ndarray X: Samples of uncorrelated points from the marginals; of shape (N,M)

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the uncorrelated samples.
        """
        if N is not None:
            distro = list()
            for i in range(len(self.D)):
                    distro1 = self.D[i].get_samples(N)

                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1)

            distro = np.reshape(distro, (len(self.D),N))
            distro = distro.T

        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method')
        return distro
    def get_correlated_samples(self, N=None):
        """
        Method for generating correlated samples.

        :param int N: Number of correlated samples required.
        :param numpy.ndarray X: Samples of correlated points from the marginals; of shape (N,M)

        :return:
            **C**: A numpy.ndarray of shape (N, M), which contains the correlated samples.
        """
        if N is not None:

            distro = list()
            for i in range(len(self.D)):
                    distro1 = self.std.get_samples(N)

                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1)

            distro = np.reshape(distro, (len(self.D),N))
            interm = np.dot(self.A, distro)
            correlated = np.zeros((len(self.D),N))
            for i in range(len(self.D)):
                for j in range(N):
                    correlated[i,j] = self.D[i].get_icdf(self.std.get_cdf(interm[i,j]))
            correlated = correlated.T
            return correlated

        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method: please choose between sampling N points or giving an array of uncorrelated data ')