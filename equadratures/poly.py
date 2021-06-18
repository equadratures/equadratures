"""The polynomial parent class; one of the main building blocks in Effective Quadratures."""
from equadratures.stats import Statistics
from equadratures.parameter import Parameter
from equadratures.basis import Basis
from equadratures.solver import Solver
from equadratures.subsampling import Subsampling
from equadratures.quadrature import Quadrature
from equadratures.datasets import score
import equadratures.plot as plot
import scipy.stats as st
import numpy as np
from copy import deepcopy
MAXIMUM_ORDER_FOR_STATS = 8
CARD_LIMIT_SOFT = int(50e3)

class Poly(object):
    """ Definition of a polynomial object.

    Parameters
    ----------
    parameters : list
        A list of parameters, where each element of the list is an instance of the Parameter class.
    basis : Basis
        An instance of the Basis class corresponding to the multi-index set used.
    method : str, optional
        The method used for computing the coefficients. Should be one of: ``compressed-sensing``, ``least-squares``, ``minimum-norm``, ``numerical-integration``, ``least-squares-with-gradients``, ``least-absolute-residual``, ``huber``, ``elastic-net``, ``relevance-vector-machine`` or ``custom-solver``. See :class:`~equadratures.solver.Solver` for further details.
    sampling_args : dict, optional
        A dict containing optional arguments centered around the specific sampling strategy:

            - **mesh** (str): Avaliable options are: ``monte-carlo``, ``sparse-grid``, ``tensor-grid``, ``induced``, or ``user-defined``. Note that when the ``sparse-grid`` option is invoked, the sparse pseudospectral approximation method [1] is the adopted. One can think of this as being the correct way to use sparse grids in the context of polynomial chaos [2] techniques.
            - **subsampling-algorithm** (str): The ``subsampling-algorithm`` input refers to the optimisation technique for subsampling. In the aforementioned four sampling strategies, we generate a logarithm factor of samples above the required amount and prune down the samples using an optimisation technique (see [1]). Existing optimisation strategies include: ``qr``, ``lu``, ``svd``, ``newton``. These refer to QR with column pivoting [2], LU with row pivoting [3], singular value decomposition with subset selection [2] and a convex relaxation via Newton's method for determinant maximization [4]. Note that if the ``tensor-grid`` option is selected, then subsampling will depend on whether the Basis argument is a total order index set, hyperbolic basis or a tensor order index set.
            - **sampling-ratio** (float): Denotes the extent of undersampling or oversampling required. For values equal to unity (default), the number of rows and columns of the associated Vandermonde-type matrix are equal.
            - **sample-points** (numpy.ndarray): A numpy ndarray with shape (number_of_observations, dimensions) that corresponds to a set of sample points over the parameter space.
            - **sample-outputs** (numpy.ndarray): A numpy ndarray with shape (number_of_observations, 1) that corresponds to model evaluations at the sample points. Note that if ``sample-points`` is provided as an input, then the code expects ``sample-outputs`` too.
            - **sample-gradients** (numpy.ndarray): A numpy ndarray with shape (number_of_observations, dimensions) that corresponds to a set of sample gradient values over the parameter space.
    solver_args : dict, optional
        Optional arguments centered around the specific solver used for computing the coefficients. See :class:`Solver<equadratures.solver.Solver>`.
    override_cardinality: bool, optional
        Option to override the soft cardinality limit of 50E3. By default the limit is enforced in order to avoid excessively long run times when the polynomial coefficients are computed.

    Examples
    --------
    >>> # Subsampling from a tensor grid
    >>> param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=3)
    >>> basis = eq.Basis('total order')
    >>> poly = eq.Poly(parameters=[param, param], basis=basis, method='least-squares' , sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'svd', 'sampling-ratio':1.0})

    >>> # User-defined data with compressive sensing
    >>> X = np.loadtxt('inputs.txt')
    >>> y = np.loadtxt('outputs.txt')
    >>> param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=3)
    >>> basis = eq.Basis('total order')
    >>> poly = eq.Poly([param, param], basis, method='compressive-sensing', sampling_args={'sample-points':X_red,'sample-outputs':Y_red})

    >>> # Using a sparse grid
    >>> param = eq.Parameter(distribution='uniform', lower=-1., upper=1., order=3)
    >>> basis = eq.Basis('sparse-grid', level=7, growth_rule='exponential')
    >>> poly = eq.Poly(parameters=[param, param], basis=basis, method='numerical-integration')

    References
    ----------
    1. Constantine, P. G., Eldred, M. S., Phipps, E. T., (2012) Sparse Pseudospectral Approximation Method. Computer Methods in Applied Mechanics and Engineering. 1-12. `Paper <https://www.sciencedirect.com/science/article/pii/S0045782512000953>`__
    2. Xiu, D., Karniadakis, G. E., (2002) The Wiener-Askey Polynomial Chaos for Stochastic Differential Equations. SIAM Journal on Scientific Computing,  24(2), `Paper <https://epubs.siam.org/doi/abs/10.1137/S1064827501387826?journalCode=sjoce3>`__
    3. Seshadri, P., Iaccarino, G., Ghisu, T., (2018) Quadrature Strategies for Constructing Polynomial Approximations. Uncertainty Modeling for Engineering Applications. Springer, Cham, 2019. 1-25. `Preprint <https://arxiv.org/pdf/1805.07296.pdf>`__
    4. Seshadri, P., Narayan, A., Sankaran M., (2017) Effectively Subsampled Quadratures for Least Squares Polynomial Approximations. SIAM/ASA Journal on Uncertainty Quantification, 5(1). `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`__
    5. Rogers, S., Girolami, M., (2016) Variability in predictions. In: A First Course in Machine Learning, Second Edition (2nd. ed.). Chapman & Hall/CRC. `Book <https://github.com/wwkenwong/book/blob/master/Simon%20Rogers%2C%20Mark%20Girolami%20A%20First%20Course%20in%20Machine%20Learning.pdf>`__
    """
    def __init__(self, parameters, basis, method=None, sampling_args=None, solver_args={}, variable=None, override_cardinality=False):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = deepcopy(basis)
        self.method = method
        self.sampling_args = sampling_args
        self.solver_args = solver_args
        self.override = override_cardinality
        self.dimensions = len(parameters)
        self.orders = []
        self.gradient_flag = False
        self.variable = variable
        for i in range(0, self.dimensions):
            self.orders.append(self.parameters[i].order)
        if not self.basis.orders :
            self.basis.set_orders(self.orders)
        # Initialize some default values!
        self.inputs = None
        self.outputs = None
        self.output_variances = None
        self.subsampling_algorithm_name = None
        self.sampling_ratio = 1.0
        self.statistics_object = None
        self.parameters_order = [ parameter.order for parameter in self.parameters]
        self.highest_order = np.max(self.parameters_order)
        if self.method is not None:
            if self.method == 'numerical-integration' or self.method == 'integration':
                self.mesh = self.basis.basis_type
                if self.basis.basis_type != 'tensor-grid' and self.basis.basis_type != 'sparse-grid' and self.basis.basis_type != 'univariate':
                    raise ValueError('tensor-grid or sparse-grid basis must be used with the numerical-integration Poly method') 
            elif self.method == 'least-squares' or self.method == 'least-absolute-residual' or self.method=='huber' or self.method=='elastic-net':
                self.mesh = 'tensor-grid'
            elif self.method == 'least-squares-with-gradients':
                self.gradient_flag = True
                self.mesh = 'tensor-grid'
            elif self.method == 'compressed-sensing' or self.method == 'compressive-sensing':
                self.mesh = 'monte-carlo'
            elif self.method == 'minimum-norm':
                self.mesh = 'monte-carlo'
            elif self.method == 'relevance-vector-machine':
                self.mesh = 'monte-carlo'
            # Now depending on user inputs, override these default values!
            sampling_args_flag = 0
            if self.sampling_args is not None:
                if 'mesh' in sampling_args:
                    self.mesh = sampling_args.get('mesh')
                    sampling_args_flag = 1
                if 'sampling-ratio' in sampling_args:
                    self.sampling_ratio = float(sampling_args.get('sampling-ratio'))
                    sampling_args_flag = 1
                if 'subsampling-algorithm' in sampling_args:
                    self.subsampling_algorithm_name = sampling_args.get('subsampling-algorithm')
                    sampling_args_flag = 1
                if 'sample-points' in sampling_args:
                    self.inputs = sampling_args.get('sample-points')
                    sampling_args_flag = 1
                    self.mesh = 'user-defined'
                if 'sample-outputs' in sampling_args:
                    self.outputs = sampling_args.get('sample-outputs')
                    sampling_args_flag = 1
                if 'sample-gradients' in sampling_args:
                    self.gradients = sampling_args.get('sample-gradients')
                    sampling_args_flag = 1
                if 'gram-schmidt-correction' in sampling_args:
                    R_Psi = sampling_args.get('gram-schmidt-correction')
                    self.inv_R_Psi = np.linalg.inv(R_Psi)
                    sampling_args_flag = 1
                if 'correlations' in sampling_args:
                    self.corr = sampling_args.get('correlations')
                    sampling_args_flag = 1
                if sampling_args_flag == 0:
                    raise ValueError( 'An input value that you have specified is likely incorrect. Sampling arguments include: mesh, sampling-ratio, subsampling-algorithm, sample-points, sample-outputs and sample-gradients.')
                # Additional optional sampling_args
                if 'sample-outputs' in sampling_args:
                    self.output_variances = sampling_args.get('sample-output-variances')

            # Check cardinality before setting points and weights etc (unless override_cardinality=True)
            if not self.override:
                cardinality = self.basis.get_cardinality()
                if cardinality >= CARD_LIMIT_SOFT:
                    raise Exception('Cardinality %.1e >= soft cardinality limit %.1e. Computing polynomial coefficients may take a long time. To override this, set override_cardinality=True.'
                        %(cardinality,CARD_LIMIT_SOFT)) 

            # Set solver, points and weight etc 
            self._set_solver()
            self._set_subsampling_algorithm()
            self._set_points_and_weights()
        else:
            print('WARNING: Method not declared.')

    def plot_polyfit_1D(self, ax=None, uncertainty=True, output_variances=None, number_of_points=200, show=True):
        """ Plots a 1D only polynomial fit to the data. See :meth:`~equadratures.plot.plot_polyfit_1D` for full description. """
        return plot.plot_polyfit_1D(self,ax,uncertainty,output_variances,number_of_points,show)

    def plot_model_vs_data(self, ax=None, sample_data=None, metric='adjusted_r2', show=True):
        """ Plots the polynomial approximation against the true data. See :meth:`~equadratures.plot.plot_model_vs_data` for full description. """
        return plot.plot_model_vs_data(self,ax,sample_data,metric,show)

    def plot_sobol(self, ax=None, order=1, show=True, labels=None, kwargs={}):
        """ Plots a polynomial's Sobol' indices of a given order. See :meth:`~equadratures.plot.plot_sobol` for full description. """
        return plot.plot_sobol(self,ax,order,show,labels,kwargs)

    def plot_parameters(self, ax=None, cols=2, show=True):
        """ Plots the probability density functions for all Parameters within a Polynomial. See :meth:`~equadratures.plot.plot_parameters` for full description. """
        return plot.plot_parameters(self, ax, cols, show)

    def plot_total_sobol(self, ax=None, show=True, labels=None, kwargs={}):
        """ Plots a polynomial's total-order Sobol' indices. See :meth:`~equadratures.plot.plot_total_sobol` for full description. """
        return plot.plot_total_sobol(self,ax,show,labels,kwargs)

    def plot_sobol_heatmap(self,parameters=None,show=True,ax=None):
        """ Generates a heatmap showing the first and second order Sobol indices. See :meth:`~equadratures.plot.plot_sobol_heatmap` for full description. """
        return plot.plot_sobol_heatmap(self,parameters,show,ax)

    def _set_parameters(self, parameters):
        """ Private function that sets the parameters. Required by the Correlated class. """
        self.parameters = parameters
        self._set_points_and_weights()

    def get_parameters(self):
        """ Returns the list of parameters.

        Returns
        -------
        list
            Contains the :class:`~equadratures.parameter.Parameter` objects belonging to the polynomial.
        """
        return self.parameters

    def get_summary(self, filename=None, tosay=False):
        """ A simple utility that writes a file summarising what the polynomial approximation has determined.

        Parameters
        ----------
        filename : str, optional
            The filename to write to. If ``None``, the summary is written to *effective-quadratures-output.txt*.
        tosay : bool, optional
            True will replace "-" signs with "minus" when writing to file for compatibility with os.say().
        """
        prec = '{:.3g}'
        if self.dimensions == 1:
            parameter_string = str('parameter.')
        else:
            parameter_string = str('parameters.')
        introduction = str('Your problem has been defined by '+str(self.dimensions)+' '+parameter_string)
        added = str('Their distributions are given as follows:')
        for i in range(0, self.dimensions):
            added_new = ('\nParameter '+str(i+1)+' '+str(self.parameters[i].get_description()))
            if i == 0:
                if self.variable is not None:
                    title = str('This polynomial concerns the output variable '+str(self.variable) + '.\n')
                    added = title + introduction + added_new
                else:
                    added = introduction + added_new
            else:
                added = added + added_new
        if self.statistics_object is not None:
            mean_value, var_value = self.get_mean_and_variance()
            X = self.get_points()
            y_eval = self.get_polyfit(X)
            y_valid = self._model_evaluations
            a,b,r,_,_ = st.linregress(y_eval.flatten(),y_valid.flatten())
            r2 = r**2
            statistics = '\n \nA summary of computed output statistics is given below:\nThe mean is estimated to be '+ prec.format(mean_value) +\
                ' while the variance is ' + prec.format(var_value) +'.\nFor the data avaliable, the polynomial approximation had a r square value of '+prec.format(r2)+'.'
            if self.dimensions > 1:
                sobol_indices_array = np.argsort(self.get_total_sobol_indices())
                final_value = sobol_indices_array[-1] + 1
                statistics_extra = str('\nAdditionally, the most important parameter--based on the total Sobol indices--was found to be parameter '+str(final_value)+'.')
                statistics = statistics + statistics_extra
            added = added + statistics
            if(tosay is True):
                added = added.replace('e-','e minus')
                added = added.replace('minus0','minus')
        if filename is None:
            filename = 'effective-quadratures-output.txt'
        output_file = open(filename, 'w')
        output_file.write(added)
        output_file.close()

    def _set_subsampling_algorithm(self):
        """ Private function that sets the subsampling algorithm based on the user-defined method. """
        polysubsampling = Subsampling(self.subsampling_algorithm_name)
        self.subsampling_algorithm_function = polysubsampling.get_subsampling_method()

    def _set_solver(self):
        """ Private function that sets the solver depending on the user-defined method. """
        self.solver = Solver.select_solver(self.method, self.solver_args)
        if self.method.lower()=="elastic-net":
            self.solver.elements=self.basis.elements
        
    def _set_points_and_weights(self):
        """ Private function that sets the quadrature points. """
        if hasattr(self, 'corr'):
            corr = self.corr
        else:
            corr = None
        self.quadrature = Quadrature(parameters=self.parameters, basis=self.basis, \
                        points=self.inputs, mesh=self.mesh, corr=corr)
        quadrature_points, quadrature_weights = self.quadrature.get_points_and_weights()
        if self.subsampling_algorithm_name is not None:
            P = self.get_poly(quadrature_points)
            W = np.mat( np.diag(np.sqrt(quadrature_weights)))
            A = W * P.T
            self.A = A
            self.P = P
            mm, nn = A.shape
            m_refined = int(np.round(self.sampling_ratio * nn))
            z = self.subsampling_algorithm_function(A, m_refined)
            self._quadrature_points = quadrature_points[z,:]
            self._quadrature_weights = quadrature_weights[z] / np.sum(quadrature_weights[z])
        else:
            self._quadrature_points = quadrature_points
            self._quadrature_weights = quadrature_weights
            P = self.get_poly(quadrature_points)
            W = np.mat( np.diag(np.sqrt(quadrature_weights)))
            A = W * P.T
            self.A = A
            self.P = P

    def get_model_evaluations(self):
        """ Returns the output model evaluations used to fit the polynomial.

        Returns
        -------
        numpy.ndarray
            Array containing the output model evaluations.
        """
        return self._model_evaluations

    def get_mean_and_variance(self):
        """ Computes the mean and variance of the model.

        Returns
        -------
        tuple
            Tuple containing two floats; the approximated mean and variance from the polynomial fit.
        """
        self._set_statistics()
        return self.statistics_object.get_mean(), self.statistics_object.get_variance()

    def get_skewness_and_kurtosis(self):
        """ Computes the skewness and kurtosis of the model.

        Returns
        -------
        tuple
            Tuple containing two floats; the approximated skewness and kurtosis from the polynomial fit.
        """
        self._set_statistics()
        return self.statistics_object.get_skewness(), self.statistics_object.get_kurtosis()

    def _set_statistics(self):
        """ Private method that is used within the statistics routines. """
        if self.statistics_object is None:
            if hasattr(self, 'inv_R_Psi'):
                # quad_pts, quad_wts = self.quadrature.get_points_and_weights()
                N_quad = 20000
                quad_pts = self.corr.get_correlated_samples(N=N_quad)
                quad_wts = 1.0 / N_quad * np.ones(N_quad)
                poly_vandermonde_matrix = self.get_poly(quad_pts)
            elif self.method != 'numerical-integration' and self.dimensions <= 6 and self.highest_order <= MAXIMUM_ORDER_FOR_STATS:
                quad = Quadrature(parameters=self.parameters, basis=Basis('tensor-grid', orders= np.array(self.parameters_order) + 1), \
                    mesh='tensor-grid', points=None)
                quad_pts, quad_wts = quad.get_points_and_weights()
                poly_vandermonde_matrix = self.get_poly(quad_pts)
            elif self.mesh == 'monte-carlo':
                quad = Quadrature(parameters=self.parameters,
                                  basis=self.basis, points=None, mesh=self.mesh, oversampling=10.0)
                quad_pts, quad_wts = quad.get_points_and_weights()
                N_quad = len(quad_wts)
                quad_wts = 1.0 / N_quad * np.ones(N_quad)
                poly_vandermonde_matrix = self.get_poly(quad_pts)
            else:
                poly_vandermonde_matrix = self.get_poly(self._quadrature_points)
                quad_pts, quad_wts = self.get_points_and_weights()

            if self.highest_order <= MAXIMUM_ORDER_FOR_STATS and (self.basis.basis_type.lower() == 'total-order'
                or self.basis.basis_type.lower() == 'hyperbolic-basis'):
                self.statistics_object = Statistics(self.parameters, self.basis,  self.coefficients,  quad_pts, \
                        quad_wts, poly_vandermonde_matrix, max_sobol_order=self.highest_order)
            else:
                self.statistics_object = Statistics(self.parameters, self.basis,  self.coefficients,  quad_pts, \
                        quad_wts, poly_vandermonde_matrix, max_sobol_order=MAXIMUM_ORDER_FOR_STATS)

    def get_sobol_indices(self, order):
        """ Computes the Sobol' indices.

        Parameters
        ----------
        order : int
            The order of the Sobol' indices required.

        Returns
        -------
        dict
            A dict comprising of Sobol' indices and constitutent mixed orders of the parameters.
        """
        self._set_statistics()
        return self.statistics_object.get_sobol(order)

    def get_total_sobol_indices(self):
        """ Computes the total Sobol' indices.

        Returns
        -------
        numpy.ndarray
            Array of length d (number of input parameters) of total Sobol' indices.
        """
        self._set_statistics()
        return self.statistics_object.get_sobol_total()

    def get_conditional_skewness_indices(self, order):
        """ Computes the skewness indices.

        Parameters
        ----------
        order : int
            The highest order of the skewness indices required.

        Returns
        -------
        dict
            A dict comprising of skewness indices and constitutent mixed orders of the parameters.
        """
        self._set_statistics()
        return self.statistics_object.get_conditional_skewness(order)

    def get_conditional_kurtosis_indices(self, order):
        """ Computes the kurtosis indices.

        Parameters
        ----------
        order : int
            The highest order of the kurtosis indices required.

        Returns
        -------
        dict
            A dict comprising of kurtosis indices and constitutent mixed orders of the parameters.
        """
        self._set_statistics()
        return self.statistics_object.get_conditional_kurtosis(order)

    def set_model(self, model=None, model_grads=None):
        """ Computes the coefficients of the polynomial via the method selected. 

        If model evaluations and/or gradients have not yet been provided to Poly via ``sample-outputs`` and/or ``sample-gradients``, they can be given here via the ``model`` and ``model_grads`` arguments.

        Parameters
        ----------
        model : ~collections.abc.Callable,numpy.ndarray
            The function that needs to be approximated. In the absence of a callable function, the input can be a ndarray model evaluations at the quadrature points.
        model_grads : ~collections.abc.Callable,numpy.ndarray
            The gradient of the function that needs to be approximated. In the absence of a callable gradient function, the input can be a ndarray of gradient evaluations at the quadrature points.

        Examples
        --------
        Defining a simple univariate example
            >>> x = np.linspace(-2,2,20).reshape(-1,1)
            >>> f = lambda x: x**2 - 1
            >>> param = eq.Parameter(lower=-2,upper=2,order=2)
            >>> basis = eq.Basis('univariate')

        Least-squares regression of existing data
            >>> y = f(x)
            >>> poly = eq.Poly(param,basis,method='least-squares',sampling_args={'mesh': 'user-defined','sample-points':x,'sample-outputs':y})
            >>> poly.set_model()
            >>> eq.datasets.score(f(xtest),poly.get_polyfit(xtest),metric='rmse')
            5.438959822042073e-16

        Numerical integration with model provided as a callable function
            >>> poly = eq.Poly(param,basis,method='numerical-integration')
            >>> poly.set_model(f)
            >>> eq.datasets.score(f(xtest),poly.get_polyfit(xtest),metric='rmse')
            5.363652779335998e-15

        Numerical integration, manually providing model evaluations at quadrature points
            >>> poly = eq.Poly(param,basis,method='numerical-integration')
            >>> x = poly.get_points()
            >>> y = f(x)
            >>> poly.set_model(y) 
            >>> eq.datasets.score(f(xtest),poly.get_polyfit(xtest),metric='rmse')
            5.363652779335998e-15

        Raises
        ------
        ValueError
            model values should be a column vector.
        """
        if (model is None) and (self.outputs is not None):
            self._model_evaluations = self.outputs
        else:
            if callable(model):
                y = evaluate_model(self._quadrature_points, model)
            else:
                y = model
                # TODO: This error gives messages that are usually not clear
                assert(y.shape[0] == self._quadrature_points.shape[0])
            if y.shape[1] != 1:
                raise ValueError( 'model values should be a column vector.')
            self._model_evaluations = y
            if self.gradient_flag:
                if (model_grads is None) and (self.gradients is not None):
                    grad_values = self.gradients
                else:
                    if callable(model_grads):
                        grad_values = evaluate_model_gradients(self._quadrature_points, model_grads, 'matrix')
                    else:
                        grad_values = model_grads
                p, q = grad_values.shape
                self._gradient_evaluations = np.zeros((p*q,1))
                W = np.diag(np.sqrt(self._quadrature_weights))
                counter = 0
                for j in range(0,q):
                    for i in range(0,p):
                        self._gradient_evaluations[counter] = W[i,i] * grad_values[i,j]
                        counter = counter + 1
                del grad_values
        self.statistics_object = None
        self._set_coefficients()

    def _set_coefficients(self, user_defined_coefficients=None):
        """ Computes the polynomial approximation coefficients.

        Parameters
        ----------
        user_defined_coefficients : numpy.ndarray, optional
            A numpy.ndarray of shape (N, 1) where N corresponds to the N coefficients provided by the user
        """
        # Check to ensure that if there any NaNs, a different basis must be used and solver must be changed
        # to least squares!
        if user_defined_coefficients is not None:
            self.coefficients = user_defined_coefficients
            return
        indices_with_nans = np.argwhere(np.isnan(self._model_evaluations))[:,0]
        if len(indices_with_nans) != 0:
            print('WARNING: One or more of your model evaluations have resulted in an NaN. We found '+str(len(indices_with_nans))+' NaNs out of '+str(len(self._model_evaluations))+'.')
            print('The code will now use a least-squares technique that will ignore input-output pairs of your model that have NaNs. This will likely compromise computed statistics.')
            self.inputs = np.delete(self._quadrature_points, indices_with_nans, axis=0)
            self.outputs = np.delete(self._model_evaluations, indices_with_nans, axis=0)
            self.subsampling_algorithm_name = None
            number_of_basis_to_prune_down = self.basis.cardinality - len(self.outputs)
            if number_of_basis_to_prune_down > 0:
                self.basis.prune(number_of_basis_to_prune_down + self.dimensions) # To make it an over-determined system!
            self.method = 'least-squares'
            self.mesh = 'user-defined'
            self._set_solver()
            self._set_points_and_weights()
            self.set_model()
        if self.mesh == 'sparse-grid':
            counter = 0
            multi_index = []
            coefficients = np.empty([1])
            multindices = np.empty([1, self.dimensions])
            for tensor in self.quadrature.list:
                P = self.get_poly(tensor.points, tensor.basis.elements)
                W = np.diag(np.sqrt(tensor.weights))
                A = np.dot(W , P.T)
                _, _ , counts = np.unique( np.vstack( [tensor.points, self._quadrature_points]), axis=0, return_index=True, return_counts=True)
                indices = [i for i in range(0, len(counts)) if  counts[i] == 2]
                b = np.dot(W , self._model_evaluations[indices])
                del counts, indices
                coefficients_i = self.solver.get_coefficients(A, b)  * self.quadrature.sparse_weights[counter]
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
            P = self.get_poly(self._quadrature_points)
            W = np.diag(np.sqrt(self._quadrature_weights))
            A = np.dot(W , P.T)
            b = np.dot(W , self._model_evaluations)
            if self.gradient_flag:
                # Now, we can reduce the number of rows!
                dP = self.get_poly_grad(self._quadrature_points)
                C = cell2matrix(dP, W)
                G = np.vstack([A, C])
                r =  np.linalg.matrix_rank(G)
                m, n = A. shape
                print('Gradient computation: The rank of the stacked matrix is '+str(r)+'.')
                print('The number of unknown basis terms is '+str(n))
                if n > r:
                    print('WARNING: Please increase the number of samples; one way to do this would be to increase the sampling-ratio.')
                self.coefficients = self.solver.get_coefficients(A, b, C, self._gradient_evaluations)
            else:
                self.coefficients = self.solver.get_coefficients(A, b)

    def get_multi_index(self):
        """ Returns the multi-index set of the basis.

        Returns
        -------
        numpy.ndarray
            Array of the coefficients with size (cardinality_of_basis, dimensions).
        """
        return self.basis.elements

    def get_coefficients(self):
        """ Returns the coefficients of the polynomial approximation.

        Returns
        -------
        numpy.ndarray 
            The coefficients with size (number_of_coefficients, 1).
        """
        return self.coefficients

    def get_points(self):
        """ Returns the samples based on the sampling strategy.

        Returns
        -------
        numpy.ndarray 
            The sampled quadrature points with shape (number_of_samples, dimension).
        """
        return self._quadrature_points

    def get_weights(self):
        """ Computes quadrature weights.

        Returns
        -------
        numpy.ndarray
            Array of the corresponding quadrature weights with shape (number_of_samples, 1).
        """
        return self._quadrature_weights

    def get_points_and_weights(self):
        """ Returns the samples and weights based on the sampling strategy.

        Returns
        -------
        tuple
            Tuple containing two ndarrays; the sampled quadrature points with shape (number_of_samples, dimension), and the corresponding quadrature weights with shape (number_of_samples, 1).
        """
        return self._quadrature_points, self._quadrature_weights

    def get_polyfit(self, stack_of_points, uq=False):
        """ Evaluates the /polynomial approximation of a function (or model data) at prescribed points.
        
        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit must be evaluated at.
        uq : bool, optional
            If ``True``, the estimated uncertainty (standard deviation) of the polynomial approximation is also returned (see [5]).

        Returns
        -------
        numpy.ndarray
            Array with shape (1, number_of_observations) corresponding to the polynomial approximation of the model.
        """
        N = len(self.coefficients)
        if uq:
            return np.dot(self.get_poly(stack_of_points).T , self.coefficients.reshape(N, 1)), self._get_polystd(stack_of_points)
        else:
            return np.dot(self.get_poly(stack_of_points).T , self.coefficients.reshape(N, 1))

    def get_polyfit_grad(self, stack_of_points, dim_index = None):
        """ Evaluates the gradient of the polynomial approximation of a function (or model data) at prescribed points.

        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit approximation's gradient must be evaluated at.
        dim_index : int, optional
            Index of dimension to evaluate gradient for.

        Returns
        -------
        numpy.ndarray
            Array of shape (dimensions, number_of_observations) corresponding to the polynomial gradient approximation of the model.
        """
        N = len(self.coefficients)
        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
        H = self.get_poly_grad(stack_of_points, dim_index=dim_index)
        grads = np.zeros((self.dimensions, no_of_points ) )
        if self.dimensions == 1:
            return np.dot(self.coefficients.reshape(N,),  H)
        for i in range(0, self.dimensions):
            grads[i,:] = np.dot(self.coefficients.reshape(N,) , H[i] )
        return grads

    def get_polyfit_hess(self, stack_of_points):
        """ Evaluates the hessian of the polynomial approximation of a function (or model data) at prescribed points.

        Parameters
        --------_-
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial fit approximation's Hessian must be evaluated at.

        Returns
        -------
        numpy.ndarray
            Array of shape (dimensions, dimensions, number_of_observations) corresponding to the polynomial Hessian approximation of the model.
        """
        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
        H = self.get_poly_hess(stack_of_points)
        if self.dimensions == 1:
            return np.dot(self.coefficients.T , H)
        hess = np.zeros((self.dimensions, self.dimensions, no_of_points))
        for i in range(0, self.dimensions):
            for j in range(0, self.dimensions):
                hess[i, j, :] = np.dot(self.coefficients.T , H[i * self.dimensions + j])
        return hess

    def get_polyfit_function(self):
        """ Returns a callable polynomial approximation of a function (or model data).

        Returns
        -------
        ~collections.abc.Callable
            A callable function.
        """
        N = len(self.coefficients)
        return lambda x: np.dot( self.get_poly(x).T ,  self.coefficients.reshape(N, 1) )

    def get_polyfit_grad_function(self):
        """ Returns a callable for the gradients of the polynomial approximation of a function (or model data).

        Returns
        -------
        ~collections.abc.Callable
            A callable function.
        """
        return lambda x : self.get_polyfit_grad(x)

    def get_polyfit_hess_function(self):
        """ Returns a callable for the hessian of the polynomial approximation of a function (or model data).

        Returns
        -------
        ~collections.abc.Callable
            A callable function.
        """
        return lambda x : self.get_polyfit_hess(x)

    def get_poly(self, stack_of_points, custom_multi_index=None):
        """ Evaluates the value of each polynomial basis function at a set of points.

        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number of observations, dimensions) at which the polynomial must be evaluated.
        custom_multi_index : numpy.ndarray, optional
            Array containing a custom multi-index set, in the format given by :meth:`~equadratures.basis.Basis.get_elements`.
        
        Returns
        -------
        numpy.ndarray
            Array of shape (cardinality, number_of_observations) corresponding to the polynomial basis function evaluations at the stack_of_points.
        """
        if custom_multi_index is None:
            basis = self.basis.elements
        else:
            basis = custom_multi_index
        basis_entries, dimensions = basis.shape

        if stack_of_points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, _ = stack_of_points.shape
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
        if hasattr(self, 'inv_R_Psi'):
            polynomial = self.inv_R_Psi.T @ polynomial
        return polynomial

    def get_poly_grad(self, stack_of_points, dim_index = None):
        """ Evaluates the gradient for each of the polynomial basis functions at a set of points, with respect to each input variable.

        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.
        dim_index : int, optional
            Index of the dimension to evaluate the gradient for. 

        Returns
        -------
        list
            A list with d elements, where d corresponds to the dimension of the problem. Each element is a numpy.ndarray of shape (cardinality, number_of_observations) corresponding to the gradient polynomial evaluations at the stack_of_points.
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
                if hasattr(self, 'inv_R_Psi'):
                    polynomialgradient = self.inv_R_Psi.T @ polynomialgradient
                R.append(polynomialgradient)
        return R

    def get_poly_hess(self, stack_of_points):
        """ Evaluates the Hessian for each of the polynomial basis functions at a set of points, with respect to each input variable.

        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the Hessian must be evaluated.

        Returns
        -------
        list
            A list with d^2 elements, where d corresponds to the dimension of the model. Each element is a numpy.ndarray of shape (cardinality, number_of_observations) corresponding to the hessian polynomial evaluations at the stack_of_points.
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
                if hasattr(self, 'inv_R_Psi'):
                    polynomialhessian = self.inv_R_Psi.T @ polynomialhessian
                H.append(polynomialhessian)

        return H
    def get_polyscore(self,X_test=None,y_test=None,metric='adjusted_r2'):
        """ Evaluates the accuracy of the polynomial approximation using the selected accuracy metric. 

        Training accuracy is evaluated on the data used for fitting the polynomial. Testing accuracy is evaluated on new data if it is provided by the ``X_test`` and ``y_test`` arguments (both must be provided together).

        Parameters
        ----------
        X_test : numpy.ndarray, optional
            An ndarray with shape (number_of_observations, dimensions), containing new data ``X_test`` data (optional).
        y_test : numpy.ndarray, optional
            An ndarray with shape (number_of_observations, 1) containing new ``y_test`` data (optional).
        metric : str, optional
            An optional string containing the scoring metric to use. Avaliable options are: ``adjusted_r2``, ``r2``, ``mae``, ``rmse``, or ``normalised_mae`` (default: ``adjusted_r2``).

        Returns
        -------
        float
            If ``X_test`` and ``y_test`` are ``None``. The training score of the model, output as a float.
        tuple
            If ``X_test`` and ``y_test`` are given. Tuple containing the train and test scores of the model.
        """
        X = self.get_points()
        y_pred = self.get_polyfit(X)
        train_score = score(self._model_evaluations, y_pred,metric, X=X)
        if X_test is not None and y_test is not None:
            y_pred_test = self.get_polyfit(X_test)
            test_score = score(y_test,y_pred_test,metric,X=X_test)
            return train_score, test_score
        else:
            return train_score

    def _get_polystd(self, stack_of_points):
        """ Private function to evaluate the uncertainty of the polynomial approximation at prescribed points, following the approach from [5].

        Parameters
        ----------
        stack_of_points : numpy.ndarray
            An ndarray with shape (number_of_observations, dimensions) at which the polynomial variance must be evaluated at.

        Returns
        -------
        numpy.ndarray
            Array of shape (number_of_observations,1) corresponding to the uncertainty (one standard deviation) of the polynomial approximation at each point.
        """
        # Training data
        X_train = self._quadrature_points
        y_train = self._model_evaluations

        # Define covariance matrix - TODO: allow non-diagonal matrix?
        # Empirical variance
        if self.output_variances is None:
            mse = ((y_train - self.get_polyfit(X_train))**2).mean()
            data_variance = np.full(X_train.shape[0],mse)
        # User defined variance (scalar)
        elif np.isscalar(self.output_variances):
            data_variance = np.full(X_train.shape[0],self.output_variances)
        # User defined variance (array)
        else:
            data_variance = self.output_variances
        Sigma = np.diag(data_variance)

        # Construct Q, the pseudoinverse of the weighted orthogonal polynomial matrix P
        P = self.get_poly(self._quadrature_points)
        W = np.diag(np.sqrt(self._quadrature_weights))
        A = np.dot(W, P.T)
        Q = np.dot( _inv( np.dot(A.T, A) ), A.T)

        # Construct A matrix for test points, but omit weights
        X_test = stack_of_points
        Po = self.get_poly(X_test)
        Ao = Po.T

        # Propagate the uncertainties
        Sigma_X = np.dot( np.dot(Q, Sigma), Q.T)
        Sigma_F = np.dot( np.dot(Ao, Sigma_X), Ao.T)
        std_F = np.sqrt( np.diag(Sigma_F) )
        return std_F.reshape(-1,1)

def _inv(M):
    """
    Private function to compute inverse of matrix M, where M is a numpy.ndarray.
    """
    ll, mm = M.shape
    M2 = M + 1e-10 * np.eye(ll)
    L = np.linalg.cholesky(M2)
    inv_L = np.linalg.inv(L)
    inv_M = inv_L.T @ inv_L
    return inv_M

def evaluate_model_gradients(points, fungrad, format='matrix'):
    """ Evaluates the model gradient at given values.
    
    Parameters
    ----------
    points : numpy.ndarray
        An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.
    fungrad : ~collections.abc.Callable
        A callable argument for the function's gradients.
    format : str, optional
        The format in which the output is to be provided: ``matrix`` will output a numpy.ndarray of shape (number_of_observations, dimensions) with gradient values, while ``vector`` will stack all the vectors in this matrix to yield a numpy.ndarray with shape (number_of_observations x dimensions, 1).

    Returns
    -------
    numpy.ndarray
        Array of gradient evaluations.

    """
    dimensions = len(points[0,:])
    if format == 'matrix':
        grad_values = np.zeros((len(points), dimensions))
        # For loop through all the points
        for i in range(0, len(points)):
            output_from_gradient_call = fungrad(points[i,:])
            for j in range(0, dimensions):
                grad_values[i,j] = output_from_gradient_call[j]
        return grad_values
    elif format == 'vector':
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
        raise ValueError( 'evalgradients(): Format must be either matrix or vector!')

def evaluate_model(points, function):
    """
    Evaluates the model function at given values.

    Parameters
    ----------
    points : numpy.ndarray
        An ndarray with shape (number_of_observations, dimensions) at which the gradient must be evaluated.
    function : ~collections.abc.Callable
        A callable argument for the function.

    Returns
    -------
    numpy.ndarray
        Array of function evaluations.
    """
    function_values = np.zeros((len(points), 1))
    for i in range(0, len(points)):
        function_values[i,0] = function(points[i,:])
    return function_values

def vector_to_2D_grid(coefficients, index_set):
    """ Handy function that converts a vector of coefficients into a matrix based on index set values.

    Parameters
    ----------
    coefficients : numpy.ndarray
        An ndarray with shape (N, 1) where N corresponds to the number of coefficient values.
    index_set :  numpy.ndarray 
        The multi-index set of the basis.

    Returns
    -------
    tuple
        Tuple (x,y,z,max_order), containing the x,y values of the meshgrid, the coefficient values, and the highest order.
    """
    max_order = int(np.max(index_set)) + 1
    x, y = np.mgrid[0:max_order, 0:max_order]
    z = np.full(x.shape, float('NaN'))
    indices = index_set.astype(int)
    l = len(coefficients)
    coefficients = np.reshape(coefficients, (1, l))
    z[indices[:,0], indices[:,1]] = coefficients
    return x, y, z, max_order

def cell2matrix(G, W):
    dimensions = len(G)
    G0 = G[0] # Which by default has to exist!
    C0 = G0.T
    rows, cols = C0.shape
    BigC = np.zeros((dimensions*rows, cols))
    counter = 0
    for i in range(0, dimensions):
        K = np.dot(W, G[i].T)
        for j in range(0, rows):
            for k in range(0,cols):
                BigC[counter,k] = K[j,k]
            counter = counter + 1
    BigC = np.mat(BigC)
    return BigC
