"""The Distribution template."""
import equadratures.plot as plot
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
import scipy as sc 
PDF_SAMPLES = 500000

class Distribution(object):
    """
    The class defines a Distribution object. It serves as a template for all distributions.

    """
    def __init__(self, name, x_range_for_pdf, mean=None, variance=None, skewness=None, kurtosis=None, endpoints=None, lower=-np.inf, upper=np.inf, rate=None, scale=None, order=2, variable='parameter', scipyparent=None):
        self.name = name
        self.mean = mean 
        self.variance = variance 
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.lower = lower 
        self.upper = upper 
        self.x_range_for_pdf = x_range_for_pdf
        self.rate = rate 
        self.scale = scale
        self.bounds = [self.lower, self.upper]
        self.order = order
        self.variable = variable
        self.parent = scipyparent
        self.endpoints = endpoints
        self.ab = None
    def __eq__(self, second_distribution):
        """
        Returns a boolean to clarify if two distributions are the same.

        :param Distribution self:
                An instance of the Distribution class.
        :param Distribution second_distribution:
                A second instance of the Distribution class.
        """
        if self.name == second_distribution.name and \
            self.mean == second_distribution.mean and \
            self.variance == second_distribution.variance and \
            self.lower == second_distribution.lower and \
            self.upper == second_distribution.upper and \
            self.rate == self.rate and \
            self.scale == self.scale and \
            self.x_range_for_pdf == self.x_range_for_pdf:
            return True 
        else:
            False
    def get_description(self):
        """
        Returns the description of the distribution.

        :param Distribution self:
                An instance of the distribution class.
        """
        pass
    def get_cdf(self, points=None):
        """
        A uniform cumulative density function.
        :param points:
                Matrix of points which have to be evaluated
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the uniform distribution.
        """
        if points is None:
            x = self.x_range_for_pdf
            return x, self.parent.cdf(x)
        else:
            return self.parent.cdf(points)
    def get_pdf(self, points=None):
        """
        A uniform probability distribution.
        :param points:
            Matrix of points which have to be evaluated
        :param double lower:
            Lower bound of the support of the uniform distribution.
        :param double upper:
            Upper bound of the support of the uniform distribution.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the uniform distribution.
        """
        if points is None:
            x = self.x_range_for_pdf
            return x, self.parent.pdf(x)
        else:
            return self.parent.pdf(points)
    def get_icdf(self, xx):
        """
        A Uniform inverse cumulative density function.

        :param: Uniform self:
            An instance of Uniform class
        :param array xx:
            Points at which the inverse cumulative density function need to be evaluated.
        :return:
            Inverse cumulative density function values of the Uniform distribution.
        """
        return self.parent.ppf(xx)
    def get_samples(self, m = None):
        """
        Generates samples from the Uniform distribution.

        :param: uniform self:
            An instance of Uniform class
        :param: integer m:
            Number of random samples. If no provided, a default number of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is not None:
            number = m
        else:
            number = 500000
        return self.parent.rvs(size=number)
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the distribution

        :param Distribution self:
            An instance of the distribution class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the distribution.
        """
        w_pdf = self.get_pdf(self.x_range_for_pdf)
        ab = custom_recurrence_coefficients(self.x_range_for_pdf, w_pdf, order)
        return ab
    def get_jacobi_eigenvectors(self, order=None):
        """ Computes the eigenvectors of the Jacobi matrix.

        Parameters
        ----------
        order : int
            Order of the recurrence coefficients.

        Returns
        -------
        numpy.ndarray
            Array of eigenvectors.
        """
        if order is None:
            order = self.order + 1
            JacobiMat = self.get_jacobi_matrix(order)
            if order == 1:
                V = [1.0]
        else:
            D, V = sc.linalg.eigh(self.get_jacobi_matrix(order))
            idx = D.argsort()[::-1]
            eigs = D[idx]
            eigVecs = V[:, idx]
        return eigVecs
    def get_jacobi_matrix(self, order=None, ab=None):
        """ Computes the Jacobi matrix---a tridiagonal matrix of the recurrence coefficients.

        Parameters
        ----------
        order : int
            Order of the recurrence coefficients.

        Returns
        -------
        numpy.ndarray
            2D array containing the Jacobi matrix.
        """
        if order is None and ab is None:
            ab = self.get_recurrence_coefficients()
            order = self.order + 1
        elif ab is None:
            ab = self.get_recurrence_coefficients(order)
        else:
            ab = ab[0:order, :]

        order = int(order)
        # The case of order 1~
        if int(order) == 1:
            JacobiMatrix = ab[0, 0]
        # For everything else~
        else:
            JacobiMatrix = np.zeros((int(order), int(order))) # allocate space
            JacobiMatrix[0,0] = ab[0,0]
            JacobiMatrix[0,1] = np.sqrt(ab[1,1])
            k = order - 1
            for u in range(1, int(k)):
                JacobiMatrix[u,u] = ab[u,0]
                JacobiMatrix[u,u-1] = np.sqrt(ab[u,1])
                JacobiMatrix[u,u+1] = np.sqrt(ab[u+1,1])

            JacobiMatrix[order-1, order-1] = ab[order-1,0]
            JacobiMatrix[order-1, order-2] = np.sqrt(ab[order-1,1])
        return JacobiMatrix
    def _get_orthogonal_polynomial(self, points, order=None, grad=False, hess=False):
        """
        Private function that evaluates the univariate orthogonal polynomial at quadrature points.

        :param Parameter self:
            An instance of the Parameter object.
        :param numpy.ndarray points:
            Points at which the orthogonal polynomial must be evaluated.
        :param int order:
            Order up to which the orthogonal polynomial must be obtained.
        """
        if order is None:
            order = self.order + 1
        else:
            order = order + 1
        gridPoints = np.array(points, copy=False).ravel()

        if (self.ab is None):
            self.ab = self.get_recurrence_coefficients(order)
        ab = self.ab

        orthopoly = np.zeros((order, len(gridPoints)))  # create a matrix full of zeros
        derivative_orthopoly = np.zeros((order, len(gridPoints))) if (grad or hess) else None
        dderivative_orthopoly = np.zeros((order, len(gridPoints))) if hess else None

        orthopoly[0, :] = 1.0

        # Cases
        if order == 1:  # CHANGED 2/2/18
            return orthopoly, derivative_orthopoly, dderivative_orthopoly
        orthopoly[1, :] = ((gridPoints - ab[0, 0]) * orthopoly[0, :])  * (1.0) / (1.0 * np.sqrt(ab[1, 1]))
        if (grad or hess) : derivative_orthopoly[1, :] = orthopoly[0, :] / (np.sqrt(ab[1, 1]))
        if order == 2:  # CHANGED 2/2/18
            return orthopoly, derivative_orthopoly, dderivative_orthopoly

        if order >= 3:  # CHANGED 2/2/18
            for u in range(2, order):  # CHANGED 2/2/18
                # Three-term recurrence rule in action!
                orthopoly[u, :] = (((gridPoints - ab[u - 1, 0]) * orthopoly[u - 1, :]) - np.sqrt(
                    ab[u - 1, 1]) * orthopoly[u - 2, :]) / (1.0 * np.sqrt(ab[u, 1]))
            if (grad or hess):
                for u in range(2, order):  # CHANGED 2/2/18
                    # Four-term recurrence formula for derivatives of orthogonal polynomials!
                    derivative_orthopoly[u,:] = ( ((gridPoints - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
            if hess:
                for u in range(2, order):
                    # Four-term recurrence formula for second derivatives of orthogonal polynomials!
                    dderivative_orthopoly[u,:] = ( ((gridPoints - ab[u-1,0]) * dderivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * dderivative_orthopoly[u-2,:] ) +  2.0 * derivative_orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))

        return orthopoly, derivative_orthopoly, dderivative_orthopoly
    def _get_local_quadrature(self, order=None, ab=None):
        """
        Returns the 1D quadrature points and weights for the parameter. WARNING: Should not be called under normal circumstances.

        :param Parameter self:
            An instance of the Parameter class
        :param int N:
            Number of quadrature points and weights required. If order is not specified, then by default the method will return the number of points defined in the parameter itself.
        :return:
            A N-by-1 matrix that contains the quadrature points
        :return:
            A 1-by-N matrix that contains the quadrature weights
        """
        if self.endpoints.lower() =='none':
            return get_local_quadrature(self, order, ab)
        elif self.endpoints.lower() == 'lower' or self.endpoints.lower() == 'upper':
            return get_local_quadrature_radau(self, order, ab)
        elif self.endpoints.lower() == 'both':
            return get_local_quadrature_lobatto(self, order, ab)
        else:
            raise(ValueError, 'Error in endpoints specification.')
    def plot_orthogonal_polynomials(self, ax=None, order_limit=None, number_of_points=200, show=True):
        """ Plots the first few orthogonal polynomials. See :meth:`~equadratures.plot.plot_orthogonal_polynomials` for full description. """
        return plot.plot_orthogonal_polynomials(self,ax,order_limit,number_of_points,show)
    def plot_pdf(self, ax=None, data=None, show=True, lim_range=True):
        """ Plots the probability density function for a Parameter. See :meth:`~equadratures.plot.plot_pdf` for full description. """
        return plot.plot_pdf(self,ax, data, show, lim_range)
    def plot_cdf(self, ax=None, show=True, lim_range=True):
        """ Plots the cumulative density function for a Parameter. See :meth:`~equadratures.plot.plot_cdf` for full description. """
        return plot.plot_cdf(self,ax, show, lim_range)
def get_local_quadrature(self, order=None, ab=None):
    # Check for extra input argument!
    if order is None:
        order = self.order + 1
    else:
        order = order + 1
    if ab is None:
        # Get the recurrence coefficients & the jacobi matrix
        JacobiMat = self.get_jacobi_matrix(order)
        ab = self.get_recurrence_coefficients(order+1)
    else:
        ab = ab[0:order+1,:]
        JacobiMat = self.get_jacobi_matrix(order, ab)
    # If statement to handle the case where order = 1
    if order == 1:
        # Check to see whether upper and lower bound are defined:
        if np.isinf(self.lower) or np.isinf(self.upper):
            p = np.asarray(self.mean).reshape((1,1))
        else:
            #print('see below!')
            #print(self.lower, self.upper)
            p = np.asarray((self.upper - self.lower)/(2.0) + self.lower).reshape((1,1))
        w = [1.0]
    else:
        D, V = sc.linalg.eigh(JacobiMat)
        idx = D.argsort()[::-1]
        eigs = D[idx]
        eigVecs = V[:, idx]

        w = np.linspace(1,order+1,order) # create space for weights
        p = np.ones((int(order),1))
        for u in range(0, len(idx) ):
            w[u] = float(ab[0,1]) * (eigVecs[0,idx[u]]**2) # replace weights with right value
            p[u,0] = eigs[u]
            #if (p[u,0] < 1e-16) and (-1e-16 < p[u,0]):
            #    p[u,0] = np.abs(p[u,0])
        p = p[::-1]
    return p, w
def get_local_quadrature_radau(self, order=None, ab=None):
    if self.endpoints.lower() == 'lower':
        end0 = self.lower
    elif self.endpoints.lower() == 'upper':
        end0 = self.upper
    if order is None:
        order = self.order - 1
    else:
        order = order - 1
    N = order
    if ab is None:
        ab = self.get_recurrence_coefficients(order+1)
    else:
        ab = ab[0:order+1, :]
    p0 = 0.
    p1 = 1.
    for i in range(0, N+1):
        pm1 = p0
        p0 = p1
        p1 = (end0 - ab[i, 0]) * p0 - ab[i, 1]*pm1
    ab[N+1, 0] = end0 - ab[N+1, 1] * p0/p1
    return get_local_quadrature(self, order=order+1, ab=ab)
def get_local_quadrature_lobatto(self, order=None, ab=None):
    if order is None:
        order = self.order - 2
    else:
        order = order - 2
    N = order
    endl = self.lower
    endr = self.upper
    if ab is None:
        ab = self.get_recurrence_coefficients(order+2)
    else:
        ab = ab[0:order+2, :]
    p0l = 0.
    p0r = 0.
    p1l = 1.
    p1r = 1.
    for i in range(0, N+2):
        pm1l = p0l
        p0l = p1l
        pm1r = p0r
        p0r = p1r
        p1l = (endl - ab[i, 0]) * p0l - ab[i, 1] * pm1l
        p1r = (endr - ab[i, 0]) * p0r - ab[i, 1] * pm1r
    det = p1l * p0r - p1r * p0l
    ab[N+2, 0] = (endl*p1l*p0r-endr*p1r*p0l)/det
    ab[N+2, 1] = (endr - endl) * p1l * p1r/det
    return get_local_quadrature(self, order=order+2, ab=ab)
