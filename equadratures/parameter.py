"""Core class for setting the properties of a univariate parameter.

References:
    - Akil Narayan Paper on Induced Distributions `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`_.
    - Walter Gautschi resources on orthogonal polynomials
    - Tiziano's paper on adaptive polynomial expansions
"""
import numpy as np
from scipy.special import gamma, betaln
import distributions as analytical
class Parameter(object):
    """
    This class defines a univariate parameter. Below are details of its constructor.

    :param double lower:
        Lower bound for the parameter.
    :param double upper:
        Upper bound for the parameter.
    :param integer order:
        Order of the parameter.
    :param string param_type:
        The type of distribution that characterizes the parameter. Options include: `Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, `Truncated-Gaussian <https://en.wikipedia.org/wiki/Truncated_normal_distribution>`_, `Beta <https://en.wikipedia.org/wiki/Beta_distribution>`_, `Cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_, `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_, `Uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_, `Gamma <https://en.wikipedia.org/wiki/Gamma_distribution>`_, `Weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_. If no string is provided, a `Uniform` distribution is assumed. If the user provides data, and would like to generate orthogonal polynomials (and quadrature rules) based on the data, they can set this option to be Custom.
    :param double shape_parameter_A:
        Most of the aforementioned distributions are characterized by two shape parameters. For instance, in the case of a `Gaussian` (or `TruncatedGaussian`), this represents the mean. In the case of a Beta distribution this represents the alpha value. For a uniform distribution this input is not required.
    :param double shape_parameter_B:
        This is the second shape parameter that characterizes the distribution selected. In the case of a `Gaussian` or `TruncatedGaussian`, this is the variance.
    :param data:
        A numpy array with data values (x-y column format). Note this option is only invoked if the user uses the Custom param_type.
    """
    def __init__(self, order, lower=None, upper=None, param_type=None, shape_parameter_A=None, shape_parameter_B=None, data=None):
        self.order = order

        if param_type is None:
            self.param_type = 'Uniform'
        else:
            self.param_type = param_type

        if lower is None and data is None:
            if self.param_type is "Exponential":
                self.lower = 0.0
            else:
                self.lower = -1.0
        else:
            self.lower = lower

        if upper is None and data is None:
            self.upper = 1.0
        else:
            self.upper = upper

        if shape_parameter_A is None:
            self.shape_parameter_A = 0
        else:
            self.shape_parameter_A = shape_parameter_A

        if shape_parameter_B is None:
            self.shape_parameter_B = 0
        else:
            self.shape_parameter_B = shape_parameter_B

        if self.param_type == 'TruncatedGaussian' :
            if upper is None or lower is None:
                raise(ValueError, 'parameter __init__: upper and lower bounds are required for a TruncatedGaussian distribution!')

        if self.lower >= self.upper  and data is None:
            raise(ValueError, 'parameter __init__: upper bounds must be greater than lower bounds!')

        if data is not None:
            self.data = data
            if self.param_type != 'Custom':
                raise(ValueError, 'parameter __init__: if data is provided then the custom distribution must be selected!')
        self.bounds = None
    def computeMean(self):
        """
        Returns the mean of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :return:
            Mean of the parameter.

        """
        if self.param_type == "Gaussian":
            mu = self.shape_parameter_A
        elif self.param_type == "TruncatedGaussian":
            mu = self.shape_parameter_A
        elif self.param_type == "Exponential":
            mu = 1.0/self.shape_parameter_A
        elif self.param_type == "Cauchy":
            mu = self.shape_parameter_A # technically the mean is undefined!
        elif self.param_type == "Weibull":
            mu = self.shape_parameter_A * gamma(1.0 + 1.0/self.shape_parameter_B)
        elif self.param_type == "Gamma":
            mu = self.shape_parameter_A * self.shape_parameter_B
        elif self.param_type == 'Custom':
            mu = np.mean(self.getSamples)
        return mu
    def getPDF(self, N):
        """
        Returns the probability density function of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer N:
            Number of points along the x-axis.
        :return:
            A 1-by-N matrix that contains the values of the x-axis along the support of the parameter.

        """
        if self.param_type is "Gaussian":
            x, y = analytical.PDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.PDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            x, y = analytical.PDF_GammaDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            x, y = analytical.PDF_WeibullDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            x, y = analytical.PDF_CauchyDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            x, y = analytical.PDF_UniformDistribution(N, self.lower, self.upper)
        elif self.param_type is "TruncatedGaussian":
            x, y = analytical.PDF_TruncatedGaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            x, y = analytical.PDF_ExponentialDistribution(N, self.shape_parameter_A)
        elif self.param_type is "Custom":
            x, y = analytical.PDF_CustomDistribution(N, self.data)
        else:
            raise(ValueError, 'parameter getPDF(): invalid parameter type!')
        return x, y
    def getSamples(self, m=None, graph=None):
        """
        Returns samples of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer m:
            Number of random samples. If no value is provided, a default of 5e5 is assumed.
        :return:
            A N-by-1 vector that contains the samples.
        """
        if m is None:
            number_of_random_samples = 500
        else:
            number_of_random_samples = m
        uniform_samples = np.random.random((number_of_random_samples, 1))
        yy = self.getiCDF(uniform_samples)
        return yy
    def getCDF(self, N):
        """
        Returns the cumulative density function of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param integer N:
            Number of points along the x-axis.
        :return:
            A 1-by-N matrix that contains the values of the x-axis along the support of the parameter.
        :return:
            A 1-by-N matrix that contains the values of the PDF of the parameter.

        """
        if self.param_type is "Gaussian":
            x, y = analytical.CDF_GaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            x, y = analytical.CDF_BetaDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            x, y = analytical.CDF_GammaDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            x, y = analytical.CDF_WeibullDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            x, y = analytical.CDF_CauchyDistribution(N, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            x, y = analytical.CDF_UniformDistribution(N, self.lower, self.upper)
        elif self.param_type is "TruncatedGaussian":
            x, y = analytical.CDF_TruncatedGaussianDistribution(N, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            x, y = analytical.CDF_ExponentialDistribution(N, self.shape_parameter_A)
        else:
            raise(ValueError, 'parameter getCDF(): invalid parameter type!')
        return x, y
    def getiCDF(self, x):
        """
        Returns values of the inverse CDF.

        :param Parameter self:
            An instance of the Parameter class.
        :param numpy array:
            A 1-by-N array of doubles where each entry is between [0,1].
        :return:
            A 1-by-N array where each entry is the inverse CDF of input x.

        """
        if self.param_type is "Gaussian":
            y = analytical.iCDF_Gaussian(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Beta":
            y = analytical.iCDF_BetaDistribution(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Gamma":
            y = analytical.iCDF_GammaDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Weibull":
            y = analytical.iCDF_WeibullDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Cauchy":
            y = analytical.iCDF_CauchyDistribution(x, self.shape_parameter_A, self.shape_parameter_B)
        elif self.param_type is "Uniform":
            y = (x - .5) * (self.upper - self.lower) + (self.upper + self.lower)/2.0
        elif self.param_type is "TruncatedGaussian":
            y = analytical.iCDF_TruncatedGaussianDistribution(x, self.shape_parameter_A, self.shape_parameter_B, self.lower, self.upper)
        elif self.param_type is "Exponential":
            y = analytical.iCDF_ExponentialDistribution(x, self.shape_parameter_A)
        elif self.param_type is "Custom":
            y = analytical.iCDF_CustomDistribution(x, self.data)
        else:
            raise(ValueError, 'parameter getiCDF(): invalid parameter type!')
        return y
    def getRecurrenceCoefficients(self, order=None):
        """
        Returns the recurrence coefficients of the parameter.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            The number of recurrence coefficients required. By default this is the same as the number of points used when the parameter constructor is initiated.
        :return:
            An order-by-2 matrix that containts the recurrence coefficients.

        """
        return recurrence_coefficients(self, order)
    def getJacobiMatrix(self, order=None):
        """
        Returns the tridiagonal Jacobi matrix.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            The number of rows and columns of the JacobiMatrix that is required. By default, this value is set to be the same as the number of points used when the parameter constructor is initiated.
        :return:
            An order-by-order sized Jacobi tridiagonal matrix.

        """
        return jacobiMatrix(self, order)
    def getJacobiEigenvectors(self, order=None):
        """
        Returns the eigenvectors of the tridiagonal Jacobi matrix. These are used for computing quadrature rules for numerical integration.

        :param Parameter self:
            An instance of the Parameter class.
        :param int order:
            Number of eigenvectors required. This function makes the call getJacobiMatrix(order) and then computes the corresponding eigenvectors.
        :return:
            A order-by-order matrix that contains the eigenvectors of the Jacobi matrix.
        """
        return jacobiEigenvectors(self, order)
    def _getOrthoPoly(self, points, order=None):
        """
        Returns orthogonal polynomials & its derivatives, evaluated at a set of points. WARNING: Should not be called under normal circumstances, without normalization of points!

        :param Parameter self:
            An instance of the Parameter class.
        :param ndarray points:
            Points at which the orthogonal polynomial (and its derivatives) should be evaluated at.
        :param int order:
            This value of order overwrites the order defined for the constructor.
        :return:
            An order-by-k matrix where order defines the number of orthogonal polynomials that will be evaluated and k defines the points at which these points should be evaluated at.
        :return:
            An order-by-k matrix where order defines the number of derivative of the orthogonal polynomials that will be evaluated and k defines the points at which these points should be evaluated at.
        """
        return orthoPolynomial_and_derivative(self, points, order)
    def _getLocalQuadrature(self, order=None, scale=None):
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
        return getlocalquadrature(self, order, scale)
    def fastInducedJacobiDistributionSetup(self, n, data):
        # Filename = fidistinv_jacobi_setup(n, alph, bet, data)
        """
        Fast computations for inverse Jacobi distributions
        """

        ns = arange(0, n)
        fprintf('One-time setup computations: Computing Jacobi (alpha=%1.3f, beta=%1.3f) induced distribution data for...\n', alph, bet);
        display_command = 'Computations for a jacobi induced distribution for alpha=%s and beta=%s'%(self.shape_parameter_A, self.shape_parameter_B)
        print(display_command)

        if self.param_type is "Beta":
            alpha = self.shape_parameter_B - 1.0 # bug fix @ 9/6/2016
            beta = self.shape_parameter_A - 1.0
        if self.param_type is "Uniform":
            alpha = 0.0
            beta = 0.0

        #% Construct piecewise polynomial data
        for q in range(0, n):
            nn = ns[q]
            display_loop = 'For the case where n=%s'%(q)
            print(display_loop)

            x, g = getlocalquadrature(self, order=nn-1)
            ug = induced_jacobi_distribution(self, x, nn-1, alpha, beta, 50)
            ug = np.insert(ug, 0, 0.0)
            ug = np.append(ug, 1.0)

            exps = [ beta/(beta + 1.0) , alpha / (alpha + 1.0) ]
            ug, exponents = fast_induced_jacobi_distribution_setup_helper_1(ug, exps)
    def induced_jacobi_distribution(self, x, n, M=None):
        """
        Evaluates the induced Jacobi distribution.

        :param Parameter self:
            An instance of the Parameter class.
        :param array x:
            Points over which the induced distribution must be computed.
        :param int order:
            Order of the distribution. Note that this value will override the order associated with the Parameter instance.
        :return:
            The median estimate (double)
        """
        if self.param_type is "Beta":
            alph = self.shape_parameter_B - 1.0 # bug fix @ 9/6/2016
            bet = self.shape_parameter_A - 1.0
        if self.param_type is "Uniform":
            alph = 0.0
            bet = 0.0
        if len(x) == 0:
            return 
        assert((alph > -1) and (bet > -1))
        assert( all(np.abs(x[:]) <= 1) )
        assert( n >= 0 )
        A = np.floor(abs(alph)) # is an integer
        Aa = alph - A
        F = np.zeros(len(x))
        F = np.zeros((len(x), 1))
        x = np.reshape(x, (len(x), 1))
        mrs_centroid = median_approximation_jacobi(alph, bet, n);
        xreflect = x > mrs_centroid
        if len(x) != 0:
            v =  self.induced_jacobi_distribution(-x[xreflect], n,  M)
            if v is not None:
                counter = 0
                for i in range(0, len(xreflect)):
                    if bool(xreflect[i]) is True:
                        F[i] = 1.0 - v[counter]
                        counter += 1
        ab = self.getRecurrenceCoefficients(n+1)
        ab[0,1] = 1.0 # To make it a probability measure
        if n > 0:
            # Zeros of p_n
            xn, wn = self._getLocalQuadrature(n)
        # This is the (inverse) n'th root of the leading coefficient square of p_n
        # We'll use it for scaling later
        kn_factor = np.exp(-1.0/(1.0 * n+1.0) * np.sum(  np.log(ab[:,1]) , axis=0  ) )
        for xq in range(0, len(x)):
            if x[xq] == -1:
                F[xq] = 0
                continue
            if xreflect[xq]:
                continue
            # Recurrence coefficients for quadrature rule
            ab = self.getRecurrenceCoefficients(2*n+A+M+1)
            ab[0,1] = 1 # To make it a probability measure
            if n > 0:
                # Transformed
                un = (2.0/(x[xq]+1.0)) * (xn + 1.0) - 1.0
            logfactor = 0.0 # Keep this so that bet(1) always equals what it did before
            for j in range(0, n+1):
                ab = quadraticModification(ab, un[j])
                logfactor += np.log( ab[0,1] * ((x[xq]+1.0)/2.0)**2 * kn_factor)
                ab[0,1] = 1.0

            # Linear modification by factors (2 - 1/2*(u+1)*(x+1)), having root u = (3-x)/(1+x)
            root = (3-x[xq])/(1+x[xq]);
            for aq in range(0, int(A) ):
                ab = linearModification(ab, root)
                logfactor += logfactor + np.log(ab[0,1] * 1.0/2.0 * (x[xq]+1.0));
                ab[0,1] = 1.0

            # M-point Gauss quadrature for evaluation of auxilliary integral I
            u, w = self._getLocalQuadrature(M)
            I = np.dot(w ,  (2.0 - 1.0/2.0 * (u+1.) * (x[xq]+1.) )**Aa )
            F[xq] = np.exp(logfactor - alph * np.log(2.0) - betaln(bet+1.0, alph+1.0) - np.log(bet+1.0) + (bet+1)* np.log((x[xq]+1.0)/2.0)) * I
        return F
    def induced_distribution_jacobi_bisection(self, u, n, alpha, beta):
        """
        Computes the inverse of the order-n induced primitive for the Jacobi distribution
        with shape parameters alpha and beta. Uses a bisection method in conjunction with forward
        evaluation given by the induced jacobi distribution function.
        """
        assert( (all(u) >= 0) and (all(u) <=1 ) )
        assert( (alpha > -1) and (beta > -1) )
        assert( all(n >= 0) )
        x = np.zeros((len(u)))
        supp = [-1, 1]

        if len(n) == 1:
            primitive = lambda (x): self.induced_jacobi_distribution(x, n)

        ab = self.getRecurrenceCoefficients(2*n+400)
        # x = idist_inverse!
        """
        if numel(n) == 1
        %primitive = @(x) jacobi_induced_primitive(x, n, alph, bet);
        primitive = @(xx) idist_jacobi(xx, n, alph, bet);

        % Need 2*n + K coefficients, where K is the size of the Markov-Stiltjies binning procedure
        [a,b] = jacobi_recurrence(2*n + 400, alph, bet);

        x = idist_inverse(u, n, primitive, a, b, supp);

        else

        nmax = max(n(:));
        [nn, ~, bin] = histcounts(n, -0.5:(nmax+0.5));

        [a,b] = jacobi_recurrence(2*nmax + 400, alph, bet);

        for qq = 0:nmax

            flags = bin==(qq+1);

            primitive = @(xx) idist_jacobi(xx, qq, alph, bet);
            x(flags) = idist_inverse(u(flags), qq, primitive, a, b, supp);

        end

        """
        return 0
    def inverse_distribution_primitive(self, u, n, primitive, supp): 
        def markov_stiltijes_initial_guess(self, u, n, supp):
            #ab = self.getRecurrenceCoefficients(n+1)
            # To make it a probability measure
            #if n > 0:
            # Zeros of p_n
            x, w = self._getLocalQuadrature(n) # n or n+1 ?
            cd = self.getRecurrenceCoefficients(n)
            cd[0,1] = 1.0 

            for k in range(0, n):
                ab = quadraticModification(cd, x[k])
                b[0,1] = 1.0

            N = len(ab)
            y, w = self._getLocalQuadrature(N)

            if supp[1] > y[N]:
                X = np.insert(y, 0, supp[0])
                X = np.append(X, supp[1])
            else:
                X = np.insert(y, 0, supp[0]) 
                X = np.append(X, y[N]) # check that y[N] = y[end]!
                W = np.cumsum(w)
                W = np.insert(W, 0, 0.0)
            W = 1.0 / (1.0 * W[N] ) * W

            for i in len(W):
                if W[i] > 1.0:
                    W[i] = 1.0
            W[N] = 1.0

            # Histograms
            #_ , j = np.bincount(u)
            j = np.digitize(u, W)
            jleft = j
            jright = jleft + 1

            flags = (jleft == N)
            jleft[flags] = N + 1
            jright[flags] = N + 1
            intervals = [X[jleft], X[jright]]
            return intervals

        if n == 1:
            intervals = self.markov_stiltijes_initial_guess(u, n, supp)
        else:
            intervals = np.zeros((len(n), 2))
            nmax = np.max(n)
            rr = np.arange(-0.5, 0.5+nmax, 1.)
            nn, __, binvalues = np.digitize(n, )
            for qq in range(0, nmax):
                flags = binvalues == (qq + 1)
                intervals(flags) = self.markov_stiltijes_initial_guess(u[flags], qq, supp)
        
        x = np.zeros((len(u)))
        for q in range(0, len(u)):
            fun = lambda (xx): primitive(xx) - u[q]
            x[q] = fzero(fun, intervals[q]) # numpy fzero command!

                

#-----------------------------------------------------------------------------------
#
#                               PRIVATE FUNCTIONS BELOW
#
#-----------------------------------------------------------------------------------
def fast_induced_jacobi_distribution_setup_helper_1(ug, exps):
    N = len(ug)
    ug_mid = 0.5 * (ug[0:N-1] + ug[1:N])
    ug = ug.append(ug_mid)
    exponents = np.zeros((2, len(ug) - 1))

    for q in range(0, len(ug) - 1):
        if np.mod(q, 2) == 1:
            exponents[0,q] = 2.0/3.0
        else:
            exponents[1,q] = 2.0/3.0
    
    exponents[0,0] = exps[0]
    exponents[1,N-1] = exps[1]
    return ug, exponents 
def fast_induced_jacobi_distribution_setup_helper_2(ug, idistinv, exponents, M):
    xx = np.linspace(np.pi, 0, M)
    vgrid = np.cos(xx)



def median_approximation_jacobi(alpha, beta, n):
    """
    Returns an estimate for the median of the order-n Jacobi induced distribution.

    :param Parameter self:
        An instance of the Parameter class
    :param int order:
        Order of the distribution. Note that this value will override the order associated with the Parameter instance.
    :return:
        The median estimate (double)
    """
    if n > 0 :
        x0 = (beta**2 - alpha**2) / (2 * n + alpha + beta)**2
    else:
        x0 = 2.0/(1.0 + (alpha + 1.0)/(beta + 1.0))  - 1.0
    return x0
def linearModification(ab, x0):
    """
    Performs a linear modification of the orthogonal polynomial recurrence coefficients. It transforms the coefficients
    such that the new coefficients are associated with a polynomial family that is orthonormal under the weight (x - x0)**2

    :param Parameter self:
        An instance of the Parameter class
    :param double:
        The shift in the weights
    :return:
        A N-by-2 matrix that contains the modified recurrence coefficients.
    """
  
    alpha = ab[:,0]
    length_alpha = len(alpha)
    beta = ab[:,1]
    sign_value = np.sign(alpha[0] - x0)
    r = np.reshape(np.abs(evaluateRatioSuccessiveOrthoPolynomials(alpha, beta, x0, N-1)) , (length_alpha - 1, 1) )
    acorrect = np.zeros((N-1, 1))
    bcorrect = np.zeros((N-1, 1))
    ab = np.zeros((N-1, N-1))

    for i in range(0, N-1):
        acorrect[i] = np.sqrt(beta[i+1]) * 1.0 / r[i]
        bcorrect[i] = np.sqrt(beta[i+1]) * r[i]

    for i in range(1, N-1):
        acorrect[i] = acorrect[i+1] - acorrect[i]
        bcorrect[i] = bcorrect[i] * 1.0/bcorrect[i-1]

    for i in range(0, N-1):
        ab[i,1] = beta[i] * bcorrect[i]
        ab[i, 0] = alpha[i] + sign * acorrect[i]

    return ab
def quadraticModification(alphabeta, x0):
    """
    Performs a quadratic modification of the orthogonal polynomial recurrence coefficients. It transforms the coefficients
    such that the new coefficients are associated with a polynomial family that is orthonormal under the weight (x - x0)**2

    :param Parameter self:
        An instance of the Parameter class
    :param double:
        The shift in the weights
    :return:
        A N-by-2 matrix that contains the modified recurrence coefficients.
    """
    N = len(alphabeta)
    alpha = alphabeta[:,0]
    beta = alphabeta[:,1]
    C = np.reshape(  christoffelNormalizedOrthogonalPolynomials(alpha, beta, x0, N-1)  , [N, 1] )
    acorrect = np.zeros((N-2, 1))
    bcorrect = np.zeros((N-2, 1))
    ab = np.zeros((N-2, 2))
    temp1 = np.zeros((N-1, 1))
    for i in range(0, N-1):
        temp1[i] = np.sqrt(beta[i+1]) * C[i+1] * C[i] * 1.0/np.sqrt(1.0 + C[i]**2)
    temp1[0] = np.sqrt(beta[1])*C[1]
    acorrect = np.diff(temp1, axis=0)
    temp1 = 1 + C[0:N-1]**2
    for i in range(0, N-2):
        bcorrect[i] = (1.0 * temp1[i+1] ) / (1.0 *  temp1[i] )
    bcorrect[0] = (1.0 + C[1]**2) * 1.0/(C[0]**2)
    for i in range(0, N-2):
        ab[i,1] = beta[i+1] * bcorrect[i]
        ab[i,0] = alpha[i+1] + acorrect[i]
    return ab
def evaluateRatioSuccessiveOrthoPolynomials(a, b, x, N):
    # Evaluates the ratio between successive orthogonal polynomials!
    nx = len(x)
    assert (N>0), "This positive integer must be greater than 0!"
    assert (N < len(a)), "Positive integer N must be less than the number of elements in a!"
    assert (N < len(b)), "Positive integer N must be less than the number of elements in b!"
    r = np.zeros((nx, N))

    # Flatten x!
    xf = x[:]
    p0 = np.ones((nx, 1)) * 1.0/np.sqrt(b[0])
    p1 = np.ones((nx, 1))
    r1 = np.ones((nx, 1))
    r2 = np.ones((nx, 1))
    for i in range(0, nx):
        p1[i] = 1.0/np.sqrt(b[1]) * ( xf[i] - a[0] ) * p0[i]
        r1[i] = p1[i] / p0[i]
    r[:,0] = r1

    for q in range(1, N):
        for i in range(0, nx):
            r2[i] = ( xf[i] - a[q] ) - np.sqrt(b[q])/ r1[i]
            r1[i] = 1.0/np.sqrt(b[q+1]) * r2[i]
        r[:,q] = r1

    return r
def christoffelNormalizedOrthogonalPolynomials(a, b, x, N):
    # Evaluates the Christoffel normalized orthogonal getPolynomialCoefficients
    nx = len(x)
    assert N>= 0
    assert N <= len(a)
    assert N <= len(b)
    C = np.zeros((nx, N+1))
    # Initialize the polynomials!
    C[:,0] = 1.0/ ( 1.0 * np.sqrt(b[0]) )
    if N > 0:
        for k in range(0, len(x)):
            C[k,1] = 1.0 / (1.0 * np.sqrt(b[1]) ) * (x[k] - a[0])
    if N > 1:
        for k in range(0, len(x)):
            C[k,2] = 1.0 / np.sqrt(1.0 + C[k,1]**2)  * (  (x[k] - a[1]) * C[k,1] - np.sqrt(b[1]) )
            C[k,2] = C[k,2] / (1.0 * np.sqrt(b[2]) )
    if N > 2:
        for nnn in range(2, N):
            for k in range(0, len(x)):
                C[k,nnn+1] = 1.0/np.sqrt(1.0 + C[k,nnn]**2) * (  (x[k] - a[nnn]) * C[k,nnn] - np.sqrt(b[nnn]) * C[k,nnn-1] / np.sqrt(1.0 + C[k, nnn-1]**2) )
                C[k,nnn+1] = C[k,nnn+1] / np.sqrt(b[nnn+1])
    return C
def recurrence_coefficients(self, order=None):

    # Preliminaries.
    N = 8000 # no. of points for analytical distributions.
    if order  is None:
        order = self.order

    # 1. Beta distribution
    if self.param_type is "Beta":
        param_A = self.shape_parameter_B - 1 # bug fix @ 9/6/2016
        param_B = self.shape_parameter_A - 1
        if(param_B <= 0):
            raise(ValueError, 'ERROR: parameter_A (beta shape parameter A) must be greater than 1!')
        if(param_A <= 0):
            raise(ValueError, 'ERROR: parameter_B (beta shape parameter B) must be greater than 1!')
        ab =  jacobi_recurrence_coefficients_01(param_A, param_B , order)
        #alpha = self.shape_parameter_A
        #beta = self.shape_parameter_B
        #lower = self.lower
        #upper = self.upper
        #x, w = analytical.PDF_BetaDistribution(N, alpha, beta, lower, upper)
        #ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0,1]

    # 2. Uniform distribution
    elif self.param_type is "Uniform":
        self.shape_parameter_A = 0.0
        self.shape_parameter_B = 0.0
        ab =  jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, order)
        self.bounds = [-1, 1]

    elif self.param_type is "Custom":
        x, w = analytical.PDF_CustomDistribution(N, self.data)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [np.min(x), np.max(x)]
        self.upper = np.max(x)
        self.lower = np.min(x)

    # 3. Analytical Gaussian defined on [-inf, inf]
    elif self.param_type is "Gaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        x, w  = analytical.PDF_GaussianDistribution(N, mu, sigma)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [-np.inf, np.inf]

    # 4. Analytical Exponential defined on [0, inf]
    elif self.param_type is "Exponential":
        lambda_value = self.shape_parameter_A
        x, w  = analytical.PDF_ExponentialDistribution(N, lambda_value)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 5. Analytical Cauchy defined on [-inf, inf]
    elif self.param_type is "Cauchy":
        x0 = self.shape_parameter_A
        gammavalue = self.shape_parameter_B
        x, w  = analytical.PDF_CauchyDistribution(N, x0, gammavalue)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [-np.inf, np.inf]

    # 5. Analytical Gamma defined on [0, inf]
    elif self.param_type is "Gamma":
        k = self.shape_parameter_A
        theta = self.shape_parameter_B
        x, w  = analytical.PDF_GammaDistribution(N, k, theta)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 6. Analytical Weibull defined on [0, inf]
    elif self.param_type is "Weibull":
        lambda_value= self.shape_parameter_A
        k = self.shape_parameter_B
        x, w  = analytical.PDF_WeibullDistribution(N, lambda_value, k)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [0, np.inf]

    # 7. Analytical Truncated Gaussian defined on [a,b]
    elif self.param_type is "TruncatedGaussian":
        mu = self.shape_parameter_A
        sigma = np.sqrt(self.shape_parameter_B)
        a = self.lower
        b = self.upper
        x, w  = analytical.PDF_TruncatedGaussianDistribution(N, mu, sigma, a, b)
        ab = custom_recurrence_coefficients(order, x, w)
        self.bounds = [self.lower, self.upper]
    else:
        raise(ValueError, 'ERROR: parameter type is undefined. Choose from Gaussian, Uniform, Gamma, Weibull, Cauchy, Exponential, TruncatedGaussian or Beta')

    return ab
def jacobi_recurrence_coefficients(param_A, param_B, order):

    a0 = (param_B - param_A)/(param_A + param_B + 2.0)
    ab = np.zeros((int(order) + 1,2))
    b2a2 = param_B**2 - param_A**2

    if order > 0 :
        ab[0,0] = a0
        ab[0,1] = ( 2**(param_A + param_B + 1) * gamma(param_A + 1) * gamma(param_B + 1) )/( gamma(param_A + param_B + 2))

    for k in range(1,int(order) + 1):
        temp = k + 1
        ab[k,0] = b2a2/((2.0 * (temp - 1) + param_A + param_B) * (2.0 * temp + param_A + param_B))
        if(k == 1):
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B)) / ( (2 * (temp - 1) + param_A + param_B  )**2 * (2 * (temp - 1) + param_A + param_B + 1))
        else:
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B) * (temp - 1 + param_A + param_B) ) / ((2 * (temp - 1) + param_A + param_B)**2 * ( 2 *(temp -1) + param_A + param_B + 1) * (2 * (temp - 1) + param_A + param_B -1 ) )

    return ab
def jacobi_recurrence_coefficients_01(param_A, param_B, order):

    ab = np.zeros((order+1,2))
    cd = jacobi_recurrence_coefficients(param_A, param_B, order)
    N = order + 1

    for i in range(0,int(N)):
        ab[i,0] = (1 + cd[i,0])/2.0

    ab[0,1] = cd[0,1]/(2**(param_A + param_B + 1))

    for i in range(1,int(N)):
        ab[i,1] = cd[i,1]/4.0

    return ab
def hermite_recurrence_coefficients(param_A, param_B, order):

    # Allocate memory
    ab = np.zeros((order,2))
    sigma2 = param_B

    if order == 1:
        ab[0,0] = 0
        ab[0,1] = gamma(param_A + 0.5)
        return ab

    # Adapted from Walter Gatuschi
    N = order - 1
    n = range(1,N+1)
    nh = [ k / 2.0 for k in n]
    for i in range(0,N,2):
        nh[i] = nh[i] + sigma2

    # Now fill in the entries of "ab"
    for i in range(0,order):
        if i == 0:
            ab[i,1] = gamma(sigma2 + 0.5)
        else:
            ab[i,1] = nh[i-1]
    ab[0,1] = gamma(param_A + 0.5)#2.0

    return ab
def custom_recurrence_coefficients(order, x, w):

    # Allocate memory for recurrence coefficients
    order = int(order)+1
    w = w / np.sum(w)
    ab = np.zeros((order+1,2))

    # Negate "zero" components
    nonzero_indices = []
    for i in range(0, len(x)):
        if w[i] != 0:
            nonzero_indices.append(i)

    ncap = len(nonzero_indices)
    x = x[nonzero_indices] # only keep entries at the non-zero indices!
    w = w[nonzero_indices]
    s = np.sum(w)

    temp = w/s
    ab[0,0] = np.dot(x, temp.T)
    ab[0,1] = s


    if order == 1:
        return ab

    p1 = np.zeros((1, ncap))
    p2 = np.ones((1, ncap))

    for j in range(0, order):
        p0 = p1
        p1 = p2
        p2 = ( x - ab[j,0] ) * p1 - ab[j,1] * p0
        p2_squared = p2**2
        s1 = np.dot(w, p2_squared.T)
        inner = w * p2_squared
        s2 = np.dot(x, inner.T)
        ab[j+1,0] = s2/s1
        ab[j+1,1] = s1/s
        s = s1

    return ab
def jacobiMatrix(self, order=None):

    if order is None:
        ab = recurrence_coefficients(self)
        order = self.order + 1
    else:
        ab = recurrence_coefficients(self, order)

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
def getlocalquadrature(self, order=None, scale=None):

    # Check for extra input argument!
    if order is None:
        order = self.order + 1
    else:
        order = order + 1

    # Get the recurrence coefficients & the jacobi matrix
    recurrence_coeffs = recurrence_coefficients(self, order)
    JacobiMat = jacobiMatrix(self, order)

    # If statement to handle the case where order = 1
    if order == 1:

        # Check to see whether upper and lower bound are defined:
        if not self.lower or not self.upper:
            local_points = [computeMean(self)]
        else:
            local_points = [(self.upper - self.lower)/(2.0) + self.lower]
        local_weights = [1.0]
    else:
        # Compute eigenvalues & eigenvectors of Jacobi matrix
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        local_points = np.sort(D) # sort by the eigenvalues
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        w = np.linspace(1,order+1,order) # create space for weights
        p = np.ones((int(order),1))
        for u in range(0, len(i) ):
            w[u] = (V[0,i[u]]**2) # replace weights with right value
            p[u,0] = local_points[u]
            if (p[u,0] < 1e-16) and (-1e-16 < p[u,0]):
                p[u,0] = np.abs(p[u,0])
        local_weights = w
        local_points = p

    if scale==True:
        scaled_points = p.copy()
        if self.param_type=='Uniform':
            for i in range(0, len(p)):
                scaled_points[i] = 0.5* ( p[i] + 1. ) * (self.upper - self.lower) + self.lower
        if self.param_type == 'Beta':
            for i in range(0, len(p)):
                scaled_points[i] =  p[i] * (self.upper - self.lower) + self.lower
        return scaled_points, local_weights
    else:
        # Return 1D gauss points and weights
        return local_points, local_weights
def jacobiEigenvectors(self, order=None):

    if order is None:
        order = self.order + 1

    JacobiMat = jacobiMatrix(self, order)
    if order == 1:
        V = [1.0]
    else:
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        V = V[:,i]

    return V
def orthoPolynomial_and_derivative(self, points, order=None):
    if order is None:
        order = self.order + 1
    else:
        order = order + 1
    gridPoints = np.asarray(points).copy()
    ab = recurrence_coefficients(self, order)
    if self.param_type == 'Uniform':
        if (gridPoints > 1.0).any() or (gridPoints < -1.0).any():
            raise(ValueError, "Points not normalized.")
    if self.param_type == 'Beta':
        if (gridPoints > 1.0).any() or (gridPoints < 0.0).any():
            raise(ValueError, "Points not normalized.")

    orthopoly = np.zeros((order, len(gridPoints))) # create a matrix full of zeros
    derivative_orthopoly = np.zeros((order, len(gridPoints)))

    # Convert the grid points to a numpy array -- simplfy life!
    gridPointsII = np.zeros((len(gridPoints), 1))
    for u in range(0, len(gridPoints)):
        gridPointsII[u,0] = gridPoints[u]
    orthopoly[0,:] = 1.0

    # Cases
    if order == 1: #CHANGED 2/2/18
        return orthopoly, derivative_orthopoly
    orthopoly[1,:] = ((gridPointsII[:,0] - ab[0,0]) * orthopoly[0,:] ) * (1.0)/(1.0 * np.sqrt(ab[1,1]) )
    derivative_orthopoly[1,:] = orthopoly[0,:] / (np.sqrt(ab[1,1]))
    if order == 2: #CHANGED 2/2/18
        return orthopoly, derivative_orthopoly

    if order >= 3: #CHANGED 2/2/18
        for u in range(2,order): #CHANGED 2/2/18
            # Three-term recurrence rule in action!
            orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0])*orthopoly[u-1,:]) - np.sqrt(ab[u-1,1])*orthopoly[u-2,:] )/(1.0 * np.sqrt(ab[u,1]))
        for u in range(2, order): #CHANGED 2/2/18
            # Four-term recurrence formula for derivatives of orthogonal polynomials!
            derivative_orthopoly[u,:] = ( ((gridPointsII[:,0] - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
        return orthopoly, derivative_orthopoly
def main():
    data = 0
    n = 3
    po = Parameter(param_type='Uniform', lower=-1., upper=1., order=n)
    x, w = po._getLocalQuadrature()
    alpha = 0. 
    beta = 0.
    G = po.induced_jacobi_distribution(x, n, 6)
    print G

main()
