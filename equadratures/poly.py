"""The polynomial parent class; one of the main building blocks in Effective Quadratures."""
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
    :param dict args: Optional arguments centered around the specific sampling strategy and correlations within the samples.

        :string mesh: Avaliable options are: ``monte-carlo``, ``induced-sampling``, ``sparse-grid``, ``tensor-grid`` or ``user-defined``.
            Note that when the ``sparse-grid`` option is invoked, the sparse pseudospectral approximation method [1]
            is the adopted. One can think of this as being the correct way to use sparse grids in the context of polynomial chaos [2] techniques.
        :string subsampling-algorithm: The ``subsampling-algorithm`` input refers to the optimisation technique for subsampling. In the aforementioned four sampling strategies,
            we generate a logarithm factor of samples above the required amount and prune down the samples using an optimisation
            technique (see [1]). Existing optimisation strategies include: ``qr``, ``lu``, ``svd``, ``newton``. These refer to QR with column
            pivoting [2], LU with row pivoting [3], singular value decomposition with subset selection [2] and a convex relaxation
            via Newton's method for determinant maximization [4]. Note that if the ``tensor-grid`` option is selected, then subsampling will depend on whether the Basis
            argument is a total order index set, hyperbolic basis or a tensor order index set.
        :float sampling-ratio: Denotes the extent of undersampling or oversampling required. For values equal to unity (default), the number of rows
            and columns of the associated Vandermonde-type matrix are equal.
        :numpy.ndarray correlation-matrix: In the case where the inputs are correlated, a user-defined correlation matrix must be provided. This matrix
            input must be symmetric, positive-definite and be of shape (number_of_inputs, number_of_inputs).
        :numpy.ndarray sample-points: A numpy ndarray with shape (number_of_observations, dimensions) that corresponds to a set of sample points over the parameter space.
        :numpy.ndarray sample-outputs: A numpy ndarray with shape (number_of_observations, 1) that corresponds to model evaluations at the sample points. Note that
            if ``sample-points`` is provided as an input, then the code expects ``sample-outputs`` too.

    **Sample constructor initialisations**::

        import numpy as np
        from equadratures import *

        # Subsampling from a tensor grid
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('total order')
        poly = Poly(parameters=[param, param], basis=basis, method='least-squares' , args={'mesh':'tensor-grid', 'subsampling-algorithm':'svd', 'sampling-ratio':1.0})

        # User-defined data with compressive sensing
        X = np.loadtxt('inputs.txt')
        y = np.loadtxt('outputs.txt')
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('total order')
        poly = Poly([param, param], basis, method='compressive-sensing', args={'sample-points':X_red, \
                                                               'sample-outputs':Y_red})

        # Using a sparse grid
        param = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        basis = Basis('sparse-grid', level=7, growth_rule='exponential')
        poly = Poly(parameters=[param, param], basis=basis, method='numerical-integration')


    **References**
        1. Constantine, P. G., Eldred, M. S., Phipps, E. T., (2012) Sparse Pseudospectral Approximation Method. Computer Methods in Applied Mechanics and Engineering. 1-12. `Paper <https://www.sciencedirect.com/science/article/pii/S0045782512000953>`__
        2. Xiu, D., Karniadakis, G. E., (2002) The Wiener-Askey Polynomial Chaos for Stochastic Differential Equations. SIAM Journal on Scientific Computing,  24(2), `Paper <https://epubs.siam.org/doi/abs/10.1137/S1064827501387826?journalCode=sjoce3>`__
        3. Seshadri, P., Iaccarino, G., Ghisu, T., (2018) Quadrature Strategies for Constructing Polynomial Approximations. Uncertainty Modeling for Engineering Applications. Springer, Cham, 2019. 1-25. `Preprint <https://arxiv.org/pdf/1805.07296.pdf>`__
        4. Seshadri, P., Narayan, A., Sankaran M., (2017) Effectively Subsampled Quadratures for Least Squares Polynomial Approximations. SIAM/ASA Journal on Uncertainty Quantification, 5(1). `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`__
        5. Bos, L., De Marchi, S., Sommariva, A., Vianello, M., (2010) Computing Multivariate Fekete and Leja points by Numerical Linear Algebra. SIAM Journal on Numerical Analysis, 48(5). `Paper <https://epubs.siam.org/doi/abs/10.1137/090779024>`__
        6. Joshi, S., Boyd, S., (2009) Sensor Selection via Convex Optimization. IEEE Transactions on Signal Processing, 57(2). `Paper <https://ieeexplore.ieee.org/document/4663892>`__

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
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'least-squares-with-gradients':
            self.gradient_flag = 1
            self.mesh = 'tensor-grid'
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'compressed-sensing' or self.method == 'compressive-sensing':
            self.mesh = 'monte-carlo'
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        elif self.method == 'minimum-norm':
            self.mesh = 'monte-carlo'
            self.sampling_ratio = 1.0
            self.subsampling_algorithm_name = None
            self.correlation_matrix = None
            self.inputs = None
            self.outputs = None
        # Now depending on user inputs, override these default values!
        if self.args is not None:
            if 'mesh' in args: self.mesh = args.get('mesh')
            if 'sampling-ratio' in args: self.sampling_ratio = float(args.get('sampling-ratio'))
            if 'subsampling-algorithm' in args: self.subsampling_algorithm_name = args.get('subsampling-algorithm')
            if 'sample-points' in args:
                self.inputs = args.get('sample-points')
                self.mesh = 'user-defined'
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
    def __set_points_and_weights(self):
        """
        Private function that sets the quadrature points.

        :param Poly self:
            An instance of the Poly object.
        """
        self.quadrature = Quadrature(parameters=self.parameters, basis=self.basis, \
                        points=self.inputs, outputs=self.outputs, correlation = self.correlation_matrix, \
                        mesh=self.mesh)
        quadrature_points, quadrature_weights = self.quadrature.get_points_and_weights()
        if self.subsampling_algorithm_name is not None:
            P = self.get_poly(quadrature_points)
            W = np.mat( np.diag(np.sqrt(quadrature_weights)))
            A = W * P.T
            mm, nn = A.shape
            m_refined = int(np.round(self.sampling_ratio * nn))
            z = self.subsampling_algorithm_function(A, m_refined)
            self.quadrature_points = quadrature_points[z,:]
            self.quadrature_weights =  quadrature_weights[z] / np.sum(quadrature_weights[z])
        else:
            self.quadrature_points = quadrature_points
            self.quadrature_weights = quadrature_weights
    def get_mean_and_variance(self):
        """
        Computes the mean and variance of the model.

        :param Poly self:
            An instance of the Poly class.

        :return:
            **mean**: The approximated mean of the polynomial fit; output as a float.

            **variance**: The approximated variance of the polynomial fit; output as a float.

        """
        if self.statistics_object is None:
            self.statistics_object = Statistics(self.coefficients, self.basis, self.parameters)
        return self.statistics_object.mean, self.statistics_object.variance
    def get_skewness_and_kurtosis(self):
        """
        Computes the skewness and kurtosis of the model.

        :param Poly self:
            An instance of the Poly class.

        :return:
            **skewness**: The approximated skewness of the polynomial fit; output as a float.

            **kurtosis**: The approximated kurtosis of the polynomial fit; output as a float.

        """
        self.statistics_object = Statistics(self.coefficients, self.basis, self.parameters, self.quadrature_points, self.quadrature_weights)
        return self.statistics_object.skewness, self.statistics_object.kurtosis
    def get_sobol_indices(self, highest_sobol_order_to_compute=1):
        """
        Computes the Sobol' indices.

        :param Poly self:
            An instance of the Poly class.
        :param int highest_sobol_order_to_compute:
            The order of the Sobol' indices required.

        :return:
            **sobol_indices**: A dict comprising of Sobol' indices and constitutent mixed orders of the parameters.
        """
        self.statistics_object = Statistics(self.coefficients, self.basis, self.parameters, max_sobol_order=highest_sobol_order_to_compute)
        return self.statistics_object.sobol
    def set_model(self, model=None, model_grads=None):
        """
        Computes the coefficients of the polynomial via the method selected.

        :param Poly self:
            An instance of the Poly class.
        :param callable model:
            The function that needs to be approximated. In the absence of a callable function, the input can be the function evaluated at the quadrature points.
        :param callable model_grads:
            The gradient of the function that needs to be approximated. In the absence of a callable gradient function, the input can be a matrix of gradient evaluations at the quadrature points.
        """
        if (model is None) and (self.outputs is not None):
            self.model_evaluations = self.outputs
        else:
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
                coefficients = np.vstack([coefficients_i, coefficients])
                multindices = np.vstack([multindices_i, multindices])
                counter = counter +  1
            multindices = np.delete(multindices, multindices.shape[0]-1, 0)
            coefficients = np.delete(coefficients, coefficients.shape[0]-1)
            unique_indices, indices , counts = np.unique(multindices, axis=0, return_index=True, return_counts=True)
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
            **multi_indices**: A numpy.ndarray of the coefficients with size (cardinality_of_basis, dimensions).
        """
        return self.basis.elements
    def get_coefficients(self):
        """
        Returns the coefficients of the polynomial approximation.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **coefficients**: A numpy.ndarray of the coefficients with size (number_of_coefficients, 1).
        """
        return self.coefficients
    def get_points(self):
        """
        Returns the samples based on the sampling strategy.

        :param Poly self:
            An instance of the Poly object.
        :return:
            **points**: A numpy.ndarray of sampled quadrature points with shape (number_of_samples, dimension).
        """
        return self.quadrature_points
    def get_weights(self):
        """
        Computes quadrature weights.

        :param Poly self:
            An instance of the Poly class.
        :return:
            **weights**: A numpy.ndarray of the corresponding quadrature weights with shape (number_of_samples, 1).

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
        return self.get_poly(stack_of_points).T *  np.mat(self.coefficients)
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
        H = self.get_poly_grad(stack_of_points, dim_index=dim_index)
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
        H = self.get_poly_hess(stack_of_points)
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
        return lambda x: np.array(self.get_poly(x).T *  np.mat(self.coefficients))
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
            **polynomial**: A numpy.ndarray of shape (cardinality, number_of_observations) corresponding to the polynomial basis function evaluations
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
            **Gradients**: A list with d elements, where d corresponds to the dimension of the problem. Each element is a numpy.ndarray of shape
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
            **Hessian**: A list with d^2 elements, where d corresponds to the dimension of the model. Each element is a numpy.ndarray of shape
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
    Evaluates the model gradient at given values.

    :param numpy.ndarray points:
        An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.
    :param callable fungrad:
        A callable argument for the function's gradients.
    :param string format:
        The format in which the output is to be provided: ``matrix`` will output a numpy.ndarray of shape
        (number_of_observations, dimensions) with gradient values, while ``vector`` will stack all the
        vectors in this matrix to yield a numpy.ndarray with shape (number_of_observations x dimensions, 1).

    :return:
        **grad_values**: A numpy.ndarray of gradient evaluations.

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
    Evaluates the model function at given values.

    :param numpy.ndarray points:
        An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.
    :param callable fungrad:
        A callable argument for the function.

    :return:
        **function_values**: A numpy.ndarray of function evaluations.
    """
    function_values = np.zeros((len(points), 1))
    for i in range(0, len(points)):
        function_values[i,0] = function(points[i,:])
    return function_values
def vector_to_2D_grid(coefficients, index_set):
    """
    Handy function that converts a vector of coefficients into a matrix based on index set values.

    :param numpy.ndarray coefficients:
        An ndarray with shape (N, 1) where N corresponds to the number of coefficient values.
    :param numpy.ndarray index_set:
        The multi-index set of the basis.

    :return:
        **x**: A numpy.ndarray of x values of the meshgrid.

        **y**: A numpy.ndarray of y values of the meshgrid.

        **z**: A numpy.ndarray of the coefficient values.

        **max_order**: int corresponds to the highest order.
    """
    max_order = int(np.max(index_set)) + 1
    x, y = np.mgrid[0:max_order, 0:max_order]
    z = np.full(x.shape, float('NaN'))
    indices = index_set.astype(int)
    l = len(coefficients)
    coefficients = np.reshape(coefficients, (1, l))
    z[indices[:,0], indices[:,1]] = coefficients
    return x, y, z, max_order