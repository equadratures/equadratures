"""The polynomial parent class."""
from equadratures.stats import Statistics
from equadratures.parameter import Parameter
from equadratures.basis import Basis
from equadratures.solver import Solver
from equadratures.subsampling import Subsampling
from equadratures.quadrature import Quadrature
import pickle
import numpy as np
from copy import deepcopy
class Poly(object):
    """
    Definition of a polynomial object.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    :param str method: The method used for computing the coefficients. Should be one of: ``compressive-sensing``,
        ``numerical-integration``, ``least-squares`` or ``minimum-norm``.
    :param tuple samples: With the format (str, dict). Here the str argument to this input specifies the sampling strategy. Avaliable options
        are: ``monte-carlo``, ``latin-hypercube``, ``induced-sampling``, ``christoffel-sampling``, ``sparse-grid``, ``tensor-grid`` or ``user-defined``.
        The second argument to this input is a dictionary, that naturally depends on the chosen string.
        Note that ``monte-carlo``, ``latin-hypercube``, ``induced-sampling`` and ``christoffel-sampling`` are random
        sampling techniques and thus their output will vary with each instance; initialization of a random seed
        is recommended to facilitate reproducibility. The second argument to this input is a dict with the following key value structure.

        :param dict args: For ``monte-carlo``, ``latin-hypercube``, ``induced-sampling`` and ``christoffel-sampling``, the following structure
            ``{'sampling-ratio': (double), 'subsampling-option': (str), 'correlation': (numpy.ndarray)}`` should be adopted. The ``sampling-ratio``
            is the of the number of samples to the number of coefficients (cardinality of the basis) and it should be greater than 1.0 for
            ``least-squares``. The ``subsampling-option`` input refers to the optimisation technique for subsampling. In the aforementioned four sampling strategies,
            we generate a logarithm factor of samples above the required amount and prune down the samples using an optimisation
            technique (see [1]). Existing optimisation strategies include: ``qr``, ``lu``, ``svd``, ``newton``. These refer to QR with column
            pivoting [2], LU with row pivoting [3], singular value decomposition with subset selection [2] and a convex relaxation
            via Newton's method for determinant maximization [4]. Note that if the ``tensor-grid`` option is selected, then subsampling will depend on whether the Basis
            argument is a total order index set, hyperbolic basis or a tensor order index set. In the case of the latter, no subsampling will be carrried out. The final input
            argument is the correlation matrix between the input parameters. This input is a numpy.ndarray of size (number of parameters, number of parameters). Should this input
            not be provided, the parameters will be assumed to be independent.
        :param dict args: For the ``user-defined`` scenario, the dict is of the form ``{'sample-points': (numpy ndarray), 'sample-outputs': (numpy ndarray), 'correlation': None}``.
            The shape of *sample-points* will have size (observations, dimensions), while the shape of *sample-outputs* will have size (observations, 1). Once again, unless explicitly
            provided, the parameters will be assumed to be independent.

    **Sample constructor initialisations**::

        import numpy as np
        from equadratures import *

        # Subsampling from a tensor grid
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('total order')
        samples = ('tensor-grid', {'subsampling-ratio': 1.1, 'subsampling-option': 'qr'})
        poly = Poly(parameters=[param, param], basis=basis, method='least-squares', samples=samples)

        # User-defined data
        X = np.loadtxt('inputs.txt')
        y = np.loadtxt('outputs.txt')
        samples = ('user-defined', {'sample-points': X, 'sample-outputs': y})
        poly = Poly(parameters=[param, param], basis=basis, method='compressive-sensing', samples=samples)

        # Using a sparse grid
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('sparse-grid', level=7, growth_rule='exponential')
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration')

        # Using a tensor grid
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('tensor-grid')
        poly = Poly(parameters=[param, param, param], basis=basis, method='numerical-integration')

        # Other declarations:
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration') # Default: integration
        poly = Poly(parameters=[param, param, param], basis=basis, method='least-squares') # Default: subsamples form a tensor grid!
        poly = Poly(parameters=[param, param, param], basis=basis, method='compressed-sensing') # Default: randomized samples!
        poly = Poly(parameters=[param, param, param], basis=basis, method='minimum-norm') # Default: Random points!
        poly = Poly(parameters=[param, param, param], basis=basis, method='minimum-norm', correlation=mat) # correlated!

        # Least squares
        poly = Poly(parameters=[param, param, param], basis=basis, method='least-squares',
                                    {'mesh':'induced', 'sampling-ratio':1.5, 'subsampling-option'})

    **References**
        1. Seshadri, P., Iaccarino, G., Ghisu, T., (2018) Quadrature Strategies for Constructing Polynomial Approximations. Uncertainty Modeling for Engineering Applications. Springer, Cham, 2019. 1-25. `Preprint <https://arxiv.org/pdf/1805.07296.pdf>`__
        2. Seshadri, P., Narayan, A., Sankaran M., (2017) Effectively Subsampled Quadratures for Least Squares Polynomial Approximations. SIAM/ASA Journal on Uncertainty Quantification 5.1 :1003-1023. `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`__
        3. Bos, L., De Marchi, S., Sommariva, A., Vianello, M., (2010) Computing Multivariate Fekete and Leja points by Numerical Linear Algebra. SIAM Journal on Numerical Analysis, 48(5). `Paper <https://epubs.siam.org/doi/abs/10.1137/090779024>`__
        4. Joshi, S., Boyd, S., (2009) Sensor Selection via Convex Optimization. IEEE Transactions on Signal Processing, 57(2). `Paper <https://ieeexplore.ieee.org/document/4663892>`__
        5. SPAM Paper by Paul
        6. Xiu and Karndiakis paper.
    """
    def __init__(self, parameters, basis, method, args=None):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = basis
        self.method = method
        self.args = args
        self.dimensions = len(parameters)
        self.orders = []
        self.gradient_flag = 0
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        if not self.basis.orders :
            self.basis.set_orders(self.orders)
        # Initialize some default values!
        if self.method == 'numerical-integration' or self.method == 'integration':
            self.mesh = self.basis.basis_type
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'least-squares':
            self.mesh = 'tensor-grid'
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = 'qr'
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'least-squares-with-gradients':
            self.gradient_flag = 1
            self.mesh = 'tensor-grid'
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = 'qr'
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'compressed-sensing' or self.method == 'compressive-sensing':
            self.mesh = 'monte-carlo'
            self.sampling_ratio = 0.8
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'minimum-norm':
            self.mesh = 'monte-carlo'
            self.sampling_ratio = 0.8
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        # Now depending on user inputs, override these default values!
        if self.args is not None:
            if 'mesh' in args: self.mesh = args.get('mesh')
            if 'sampling-ratio' in args: self.sampling_ratio = float(args.get('sampling-ratio'))
            if 'subsampling-algorithm' in args: self.subsampling_algorithm_name = args.get('subsampling-algorithm')
            if 'sample-points' in args: self.inputs = args.get('sample-points')
            if 'sample-outputs' in args: self.outputs = args.get('sample-outputs')
            if 'correlation' in args: self.correlation_matrix = args.get('correlation')
        self.__set_solver()
        self.__set_subsampling_algorithm()
        self.__set_points_and_weights()
        self.statistics_object = None
    def __set_subsampling_algorithm(self):
        """
        Private function that sets the subsampling algorithm based on the user-defined method.

        :param Poly self:
            An instance of the Poly object.
        """
        polysubsampling = Subsampling(self.subsampling_algorithm_name)
        self.subsampling_algorithm_function = polysubsampling.get_subsampling_method()
    def __set_solver(self):
        """
        Private function that sets the solver depending on the user-defined method.

        :param Poly self:
            An instance of the Poly object.
        """
        polysolver = Solver(self.method)
        self.solver = polysolver.get_solver()
    def __get_weights(self, points):
        """
        Private function that sets the quadrature weights, when given the quadrature points.

        :param Poly self:
            An instance of the Poly object.
        """
        return self.quadrature_weights
    def __set_points_and_weights(self):
        """
        Private function that sets the quadrature points.

        :param Poly self:
            An instance of the Poly object.
        """
        # Samples
        self.quadrature = Quadrature(parameters=self.parameters, basis=self.basis, \
                        points=self.inputs, outputs=self.outputs, correlation = self.correlation_matrix, \
                        mesh=self.mesh)
        quadrature_points, quadrature_weights = self.quadrature.get_points_and_weights()
        # Subsampling
        if self.subsampling_algorithm_name is not None:
            P = self.get_poly(quadrature_points)
            W = np.mat( np.diag(np.sqrt(quadrature_weights)))
            A = W * P.T
            mm, nn = A.shape
            m_refined = int(np.round(self.sampling_ratio * nn))
            z = self.subsampling_algorithm_function(A, m_refined)
            self.quadrature_points = evaled_pts[z,:]
            self.quadrature_weights =  weights[z] / np.sum(weights[z])
        else:
            self.quadrature_points = quadrature_points
            self.quadrature_weights = quadrature_weights
    def get_mean_and_variance(self):
        """
        Computes the mean and variance of the model.

        :param Poly self:
            An instance of the Poly class.
        """
        if self.statistics_object is None:
            self.statistics_object = Statistics(self.coefficients, self.basis, self.parameters)
        return self.statistics_object.mean, self.statistics_object.variance
    def get_sobol_indices(self, highest_sobol_order_to_compute=1):
        """
        Computes the Sobol' indices.

        :param Poly self:
            An instance of the Poly class.
        :param int highest_sobol_order_to_compute:
            The order of the Sobol' indices required.
        """
        self.statistics_object = Statistics(self.coefficients, self.basis, self.parameters, max_sobol_order=highest_sobol_order_to_compute)
        return self.statistics_object.sobol
    def set_model(self, model, model_grads=None):
        """
        Computes the coefficients of the polynomial via the method selected.

        :param Poly self:
            An instance of the Poly class.
        :param callable model:
            The function that needs to be approximated. In the absence of a callable function, the input can be the function evaluated at the quadrature points.
        :param callable model_grads:
            The gradient of the function that needs to be approximated. In the absence of a callable gradient function, the input can be a matrix of gradient evaluations at the quadrature points.
        """
        # Model evaluation
        if callable(model):
            y = evaluate_model(self.quadrature_points, model)
        else:
            y = model
            assert(y.shape[0] == self.quadrature_points.shape[0])
        if y.shape[1] != 1:
            raise(ValueError, 'model values should be a column vector.')
        self.model_evaluations = y
        if self.gradient_flag == 1:
            if callable(model_grads):
                grad_values = evaluate_model_gradients(self.quadrature_points, model_grads, 'matrix')
            else:
                grad_values = model_grads
            p, q = grad_values.shape
            self.gradient_evaluations = np.zeros((p*q,1))
            counter = 0
            for j in range(0,q):
                for i in range(0,p):
                    self.gradient_evaluations[counter] = W[i,i] * grad_values[i,j]
                    counter = counter + 1
            del d, grad_values
            dP = self.get_poly_grad(self.quadrature_points)
        self.__set_coefficients()
    def __set_coefficients(self):
        """
        Computes the polynomial approximation coefficients.

        :param Poly self:
            An instance of the Poly object.
        """
        if self.mesh == 'sparse-grid':
            counter = 0
            multi_index = []
            coefficients = np.empty([1])
            multindices = np.empty([1, self.dimensions])
            for tensor in self.quadrature.list:
                P = self.get_poly(tensor.points, tensor.basis.elements)
                W = np.diag(np.sqrt(tensor.weights))
                A = np.dot(W , P.T)
                __, __ , counts = np.unique( np.vstack( [tensor.points, self.quadrature_points]), axis=0, return_index=True, return_counts=True)
                indices = [i for i in range(0, len(counts)) if  counts[i] == 2]
                b = np.dot(W , self.model_evaluations[indices])
                del counts, indices
                coefficients_i = self.solver(A, b)  * self.quadrature.sparse_weights[counter]
                multindices_i =  tensor.basis.elements
                print(coefficients_i, multindices_i)
                print('---------')
                coefficients = np.vstack([coefficients_i, coefficients])
                multindices = np.vstack([multindices_i, multindices])
                counter = counter +  1
            multindices = np.delete(multindices, multindices.shape[0]-1, 0)
            coefficients = np.delete(coefficients, multindices.shape[0]-1, 0)
            __, indices , counts = np.unique(multindices, axis=0, return_index=True, return_counts=True)
            unique_indices = multindices[indices]
            coefficients_final = np.zeros((unique_indices.shape[0], 1))
            for i in range(0, unique_indices.shape[0]):
                for j in range(0, multindices.shape[0]):
                    if np.array_equiv( unique_indices[i,:] , multindices[j,:]):
                        coefficients_final[i] = coefficients_final[i] + coefficients[j]
            self.coefficients = coefficients_final
            self.basis.elements = unique_indices
        else:
            P = self.get_poly(self.quadrature_points)
            W = np.diag(np.sqrt(self.quadrature_weights))
            A = np.dot(W , P.T)
            b = np.dot(W , self.model_evaluations)
            if self.gradient_flag == 1:
                C = cell2matrix(dPcell, W)
                self.coefficients = self.solver(A, b, C, self.gradient_evaluations)
            else:
                self.coefficients = self.solver(A, b)
    def get_multi_index(self):
        """
        Returns the multi-index set of the basis.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **c**: A numpy.ndarray of the coefficients with size (cardinality_of_basis, dimensions).
        """
        return self.basis.elements
    def get_coefficients(self):
        """
        Returns the coefficients of the polynomial approximation.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **c**: A numpy.ndarray of the coefficients with size (number_of_coefficients, 1).
        """
        return self.coefficients
    def get_points(self):
        """
        Returns the samples based on the sampling strategy.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **x**: A numpy.ndarray of sampled quadrature points with shape (number_of_samples, dimension).
        """
        return self.quadrature_points
    def get_weights(self):
        """
        Computes quadrature weights.

        :param Poly self:
            An instance of the Poly class.
        :return:
            **w**: A numpy.ndarray of the corresponding quadrature weights with shape (number_of_samples, 1).

        """
        return self.quadrature_weights
    def get_points_and_weights(self):
        """
        Returns the samples and weights based on the sampling strategy.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **x**: A numpy.ndarray of sampled quadrature points with shape (number_of_samples, dimension).

            **w**: A numpy.ndarray of the corresponding quadrature weights with shape (number_of_samples, 1).
        """
        return self.quadrature_points, self.quadrature_weights
    def get_polyfit(self, stack_of_points):
        """
        Evaluates the the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit must be evaluated at.
        :return:
            **p**: A numpy.ndarray of shape (1, number_of_observations) corresponding to the polynomial approximation of the model.
        """
        return self.getPolynomial(stack_of_points).T *  np.mat(self.coefficients)
    def get_polyfit_grad(self, stack_of_points, dim_index = None):
        """
        Evaluates the gradient of the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit approximation's
            gradient must be evaluated at.
        :return:
            **p**: A numpy.ndarray of shape (dimensions, number_of_observations) corresponding to the polynomial gradient approximation of the model.
        """
        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
        H = self.getPolynomialGradient(stack_of_points, dim_index=dim_index)
        grads = np.zeros((self.dimensions, no_of_points ) )
        if self.dimensions == 1:
            return np.mat(self.coefficients).T * H
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads
    def get_polyfit_hess(self, stack_of_points):
        """
        Evaluates the hessian of the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit approximation's
            Hessian must be evaluated at.
        :return:
            **h**: A numpy.ndarray of shape (dimensions, dimensions, number_of_observations) corresponding to the polynomial Hessian approximation of the model.
        """
        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
        H = self.getPolynomialHessian(stack_of_points)
        if self.dimensions == 1:
            return np.mat(self.coefficients).T * H
        hess = np.zeros((self.dimensions, self.dimensions, no_of_points))
        for i in range(0, self.dimensions):
            for j in range(0, self.dimensions):
                hess[i, j, :] = np.mat(self.coefficients).T * H[i * self.dimensions + j]
        return hess
    def get_polyfit_function(self):
        """
        Returns a callable polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.
        """
        return lambda x: np.array(self.getPolynomial(x).T *  np.mat(self.coefficients))
    def get_polyfit_grad_function(self):
        """
        Returns a callable for the gradients of the polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.
        """
        return lambda x : self.evaluatePolyGradFit(x)
    def get_polyfit_hess_function(self):
        """
        Returns a callable for the hessian of the polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.
        """
        return lambda x : self.evaluatePolyHessFit(x)
    def get_poly(self, stack_of_points, custom_multi_index=None):
        """
        Evaluates the value of each polynomial basis function at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number of observations, dimensions) at which the polynomial must be evaluated.

        :return:
            **p**: A numpy.ndarray of shape (cardinality, number_of_observations) corresponding to the polynomial basis function evaluations
            at the stack_of_points.
        """
        if custom_multi_index is None:
            basis = self.basis.elements
        else:
            basis = custom_multi_index
        basis_entries, dimensions = basis.shape

        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, __ = stack_of_points.shape
        p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ , _ =  self.parameters[0]._get_orthogonal_polynomial(stack_of_points, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stack_of_points.shape) == 1:
                    stack_of_points = np.array([stack_of_points])
                p[i] , _ , _ = self.parameters[i]._get_orthogonal_polynomial(stack_of_points[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        polynomial = np.ones((basis_entries, no_of_points))
        for k in range(dimensions):
            basis_entries_this_dim = basis[:, k].astype(int)
            polynomial *= p[k][basis_entries_this_dim]
        return polynomial
    def get_poly_grad(self, stack_of_points, dim_index = None):
        """
        Evaluates the gradient for each of the polynomial basis functions at a set of points,
        with respect to each input variable.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.

        :return:
            **dp**: A list with d elements, where d corresponds to the dimension of the problem. Each element is a numpy.ndarray of shape
            (cardinality, number_of_observations) corresponding to the gradient polynomial evaluations at the stack_of_points.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if len(stack_of_points.shape) == 1:
            if dimensions == 1:
                # a 1d array of inputs, and each input is 1d
                stack_of_points = np.reshape(stack_of_points, (len(stack_of_points),1))
            else:
                # a 1d array representing 1 point, in multiple dimensions!
                stack_of_points = np.array([stack_of_points])
        no_of_points, _ = stack_of_points.shape
        p = {}
        dp = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _ , dpoly, _ =  self.parameters[0]._get_orthogonal_polynomial(stack_of_points, int(np.max(basis) ) )
            return dpoly
        else:
            for i in range(0, dimensions):
                if len(stack_of_points.shape) == 1:
                    stack_of_points = np.array([stack_of_points])
                p[i] , dp[i], _ = self.parameters[i]._get_orthogonal_polynomial(stack_of_points[:,i], int(np.max(basis[:,i])) )

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
    def get_poly_hess(self, stack_of_points):
        """
        Evaluates the Hessian for each of the polynomial basis functions at a set of points,
        with respect to each input variable.

        :param Poly self:
            An instance of the Poly class.
        :param numpy.ndarray stack_of_points:
            An ndarray with shape (number_of_observations, dimensions) at which the Hessian must be evaluated.

        :return:
            **h**: A list with d^2 elements, where d corresponds to the dimension of the model. Each element is a numpy.ndarray of shape
            (cardinality, number_of_observations) corresponding to the hessian polynomial evaluations at the stack_of_points.

        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
        p = {}
        dp = {}
        d2p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _, _, d2poly = self.parameters[0]._get_orthogonal_polynomial(stack_of_points, int(np.max(basis)))
            return d2poly
        else:
            for i in range(0, dimensions):
                if len(stack_of_points.shape) == 1:
                    stack_of_points = np.array([stack_of_points])
                p[i], dp[i], d2p[i] = self.parameters[i]._get_orthogonal_polynomial(stack_of_points[:, i],
                                                                       int(np.max(basis[:, i]) + 1))
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
                            polynomialhessian[i, :] = d2p[k][int(basis[i, k])] * temp
                        elif k == gradDirection1:
                            polynomialhessian[i, :] = dp[k][int(basis[i, k])] * temp
                        elif k == gradDirection2:
                            polynomialhessian[i, :] = dp[k][int(basis[i, k])] * temp
                        else:
                            polynomialhessian[i, :] = p[k][int(basis[i, k])] * temp
                        temp = polynomialhessian[i, :]
                H.append(polynomialhessian)

        return H
def evaluate_model_gradients(points, fungrad, format):
    """
    NEED TO COMPLETE.
    """
    dimensions = len(points[0,:])
    if format is 'matrix':
        grad_values = np.zeros((len(points), dimensions))
        # For loop through all the points
        for i in range(0, len(points)):
            output_from_gradient_call = fungrad(points[i,:])
            for j in range(0, dimensions):
                grad_values[i,j] = output_from_gradient_call[j]
        return grad_values
    elif format is 'vector':
        grad_values = np.zeros((len(points) * dimensions, 1))
        # For loop through all the points
        counter = 0
        for i in range(0, len(points)):
            output_from_gradient_call = fungrad(points[i,:])
            for j in range(0, dimensions):
                grad_values[counter, 0] = output_from_gradient_call[j]
                counter = counter + 1
        return np.mat(grad_values)
    else:
        error_function('evalgradients(): Format must be either matrix or vector!')
        return 0
def evaluate_model(points, function):
    """
    NEED TO COMPLETE.
    """
    function_values = np.zeros((len(points), 1))
    # For loop through all the points
    for i in range(0, len(points)):
        function_values[i,0] = function(points[i,:])
    return function_values
def vector_to_2D_grid(coefficients, index_set):
    """
    NEED TO COMPLETE.
    """
    max_order = int(np.max(index_set)) + 1
    x, y = np.mgrid[0:max_order, 0:max_order]
    z = np.full(x.shape, float('NaN'))
    indices = index_set.astype(int)
    l = len(coefficients)
    coefficients = np.reshape(coefficients, (1, l))
    z[indices[:,0], indices[:,1]] = coefficients
    return x, y, z, max_order