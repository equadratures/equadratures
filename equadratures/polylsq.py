"""Operations involving multivariate polynomials (with gradients) via least squares"""
from parameter import Parameter
from polyint import Polyint
from qr import qr_MGS, solveLSQ, solveCLSQ
from indexset import IndexSet
from utils import evalfunction, evalgradients
from stats import Statistics
from convex import maxdet, binary2indices
import numpy as np
from scipy import linalg 

class Polylsq(object):
    """
    This class defines a Polylsq (polynomial via least squares) object

    :param array uq_parameters: A list of Parameter objects.
    :param IndexSet index_set: Polynomial index set. If an index set is not given, the constructor uses a tensor grid basis of polynomials. For total order and hyperbolic index sets, the user needs to explicity input an index set.
    :param string method: Subsampling strategy; options include: 'QR', which is the default option and 'Random'. See this `paper <https://arxiv.org/abs/1601.05470>`_, for further details. 
        We will be adding a 'Hybrid' strategy shortly that combines both QR and randomized techniques.
    
    Attributes:
        * **self.A**: (numpy matrix) Matrix of the multivariate orthogonal polynomial (defined as per the index set basis) evaluated at all points of a tensor grid.
        * **self.C**: (cell) A cell of d-dimensional numpy matrices of the derivative of the orthogonal polynomial evaluated at all points of a tensor grid.
        * **self.index_set**:(numpy array) An array of the indices used in the multivariate Polynomial. This changes when the user prunes down the number of columns.
        * **self.tensor_quadrature_points**: (numpy matrix) Full tensor grid quadrature points
        * **self.tensor_quadrature_weights**: (numpy matrix) Full tensor grid quadrature weights
        * **self.A_subsampled**: (numpy matrix) Subsampled rows of A, obtained either via QR column pivoting or randomization
        * **self.C_subsampled**: (numpy matrix) A stacked matrix comprising of subsampled rows of each matrix in the self.C
        * **self.no_of_basis_terms**: (int) The number of columns in A_subsampled
        * **self.dimension**: (int) The number of dimensions of the Polynomial
        * **self.no_of_evals**: (int) The minimum number of model evaluations required to estimate the coefficients of the Polynomial
        * **self.row_indices**: (int) The rows of A that are used in construction A_subsampled
        * **self.subsampled_quadrature_points**: (numpy matrix) Subsampled quadrature points
        * **self.subsampled_quadrature_weights**: (numpy matrix) Subsampled quadrature weights

    **Notes:** 
    
    For the exact definitions of A, C, A_subsampled and C_subsampled please see: `paper <https://arxiv.org/abs/1601.05470>`_. 

    **Sample usage:** 
    ::
        
        >> var1 = Parameter(points=12, shape_parameter_A=0.5, param_type='Exponential')
        >> I = IndexSet('Total order' [3, 3, 3])
        >> eq = Polysubs([var1, var1], I)
        >> print eq.A
        >> print eq.dimensions
        >> print eq.subsampled_quadrature_points
        >> print eq.no_of_evals
        >> print esq.C_subsampled
    """

    # Constructor
    def __init__(self, uq_parameters, index_set=None, method=None):
        self.uq_parameters = uq_parameters
        dimensions = len(uq_parameters)
        # For increased flexibility, if the index_set is not given, we will assume a tensor grid basis
        if index_set is None:
            # determine the orders!
            orders_to_use = []
            for u in range(0, dimensions):
                orders_to_use.append(np.int(uq_parameters[u].order - 1) )
            # Use the tensor grid option!
            self.index_set = IndexSet("Tensor grid", orders_to_use)    
        else:
            # Now before we set self.index_set = index_set, we check to make sure that
            # the number of basis used is -1 the number of points!
            orders_to_use = []
            count = 0
            for u in range(0, dimensions):
                orders_to_use.append( np.int(uq_parameters[u].order) )
                if orders_to_use[u] <= index_set.orders[u] :
                    count = count + 1
            if count > 0:
                raise(ValueError, 'IndexSet: Basis orders: Ensure that the basis order is always -1 the number of points!')
            self.index_set = index_set
        if method is not None:
            self.method = method
        else:
            self.method = 'QR'
        # Items to set!
        A, quadrature_pts, quadrature_wts = getA(self)
        self.A = A
        self.tensor_quadrature_points = quadrature_pts
        self.tensor_quadrature_weights = quadrature_wts
        if uq_parameters[0].derivative_flag == 1:
            self.C = getC(self)
        else:
            self.C = None
        self.no_of_basis_terms = index_set.cardinality
        self.C_subsampled = None
        self.A_subsampled = None
        self.no_of_evals = None
        self.b_subsampled = None
        self.d_subsampled = None
        self.subsampled_quadrature_points = None
        self.subsampled_quadrature_weights = None # stored as a diagonal matrix??
        self.row_indices = None
        self.dimensions = len(uq_parameters)
    
    def integrate(self, function_values, gradient_values=None, technique=None):
        """
        An integration utility using effective quadrature subsampling

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :param callable function_values: A callable function or a numpy matrix of model evaluations at the quadrature subsamples.
        :param callable gradient_values: A callable function of a numpy matrix of gradient evaluations at the quadrature subsamples.
        :param string technique: The least squares technique to be used; options include: 'weighted' (default), 'constrainedDE', 'constrainedNS'. These options only matter when using gradient evaluations. They correspond to a stacked / weighted least squares approach, a constrained approach using       direct elimination, and a constrained approach using the null space method. This function is still a work in progress! ArXiv preprint underway.
        :return: 
            * **coefficients (numpy matrix)**: Coefficients of the least squares solution.
            * **cond (double)**: Condition number of the matrix on which least squares was performed.
        """

        coefficients, cond = self.computeCoefficients(function_values, gradient_values=None, technique=None)
        integral = coefficients[0]

        # For normalizing!
        #flags = []
        #uniform = 1
        #not_uniform = 0
        #for i in range(0, len(self.uq_parameters)):
        #    if self.uq_parameters[i].param_type is 'Uniform':
        #        flags.append(uniform)
        #    else:
        #        flags.append(not_uniform)

        #for i in range(0, len(self.uq_parameters)):
        #    if flags[i] == 0:
        #        integral  = integral
        #    elif flags[i] == 1:
        #        integral = integral * (self.uq_parameters[i].upper - self.uq_parameters[i].lower )

        return integral[0]
    
    def set_no_of_evals(self, no_of_evals):
        """
        Sets the number of model evaluations the user wishes to afford for generating the polynomial. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class.
        :param integer no_of_evals: The number of subsamples the user requires

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """

        # Once the user provides the number of evaluations required, we can set a few items!
        self.no_of_evals = no_of_evals
        Asquare, esq_pts, W, row_indices = getSquareA(self)
        self.A_subsampled = Asquare
        self.subsampled_quadrature_points = esq_pts
        self.subsampled_quadrature_weights = W
        self.row_indices = row_indices

        # If the user has turned on the gradient flag!
        if self.C is not None:
            dimensions = len(self.C)
            C0 = self.C[0] # Which by default has to exist!
            C0 = C0.T
            rows, cols = C0.shape
            C_subsampled = np.mat( np.zeros((dimensions*len(row_indices), cols)), dtype='float64')
            counter = 0
            for i in range(0, dimensions):
                temp_matrix = self.C[i].T
                for j in range(0, len(row_indices)):
                    for k in range(0,cols):
                        C_subsampled[counter,k] = temp_matrix[row_indices[j],k]
                    counter = counter + 1 
            self.C_subsampled = C_subsampled

    def prune(self, number_of_columns_to_delete):  
        """
        Prunes the number of columns based on the ones with the highest total orders.  

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class.
        :param integer number_of_columns_to_delete: The number of columns that need to be deleted, which is obviously less than the total number of columns. 

        """
        A = self.A_subsampled
        m, n = A.shape
        A_pruned = A[0:m, 0 : (n - number_of_columns_to_delete)]
        self.A_subsampled = A_pruned
        self.index_set.prune(number_of_columns_to_delete)

        # If clause for gradient case!
        if self.C is not None:
            C = self.C_subsampled
            p, q = C.shape
            C_pruned = C[0:p, 0 : (q - number_of_columns_to_delete)]
            self.C_subsampled = C_pruned
            
        self.no_of_basis_terms = self.no_of_basis_terms - number_of_columns_to_delete
    
    def least_no_of_subsamples_reqd(self):
        """
        Computes the least number of subsamples required when using effectively subsampled quadratures. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class
        :return: points, The least number of subsamples required. In the absence of gradients, this function simply returns the number of basis terms. In the presence of gradients this function uses an iterative rank-determination algorithm to compute the number of subsamples required.
        :rtype: int
        """
        if self.C is None:
            return self.no_of_basis_terms
        else:
            k = 1
            self.set_no_of_evals(1)
            rank = np.linalg.matrix_rank(np.mat( np.vstack([self.A_subsampled, self.C_subsampled]), dtype='float64') )
            while rank < self.no_of_basis_terms:
                k = k + 1
                self.set_no_of_evals(k)
                rank = np.linalg.matrix_rank(np.mat( np.vstack([self.A_subsampled, self.C_subsampled]), dtype='float64') )
            return k  

    def computeCoefficients(self, function_values, gradient_values=None, technique=None):
        """
        Returns the coefficients for the effectively subsampled quadratures least squares problem. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class.
        :param callable function_values: A callable function or a numpy matrix of model evaluations at the quadrature subsamples.
        :param callable gradient_values: A callable function of a numpy matrix of gradient evaluations at the quadrature subsamples.
        :param string technique: The least squares technique to be used; options include: 'weighted' (default), 'constrainedDE', 'constrainedNS'. These options only matter when using gradient evaluations. They correspond to a stacked / weighted least squares approach, a constrained approach using       direct elimination, and a constrained approach using the null space method. This function is still a work in progress! ArXiv preprint underway.
        :return: 
            * **coefficients (numpy matrix)**: Coefficients of the least squares solution.
            * **cond (double)**: Condition number of the matrix on which least squares was performed.

        **Sample usage:** 
        For useage please see the ipython-notebooks at www.effective-quadratures.org
        """
        A, normalizations = rowNormalize(self.A_subsampled)
        
        # Check if user input is a function or a set of function values!
        if callable(function_values):
            fun_values = evalfunction(self.subsampled_quadrature_points, function_values)
        else:
            fun_values = function_values
        
        
        b = self.subsampled_quadrature_weights * fun_values
        self.b_subsampled = b
        b = np.dot(normalizations, b)
        
        ################################
        # No gradient case!
        ################################
        if gradient_values is None:
            x, cond = solveLSQ(A, b)
        
        ################################
        # Gradient case!
        ################################
        else:
            if callable(gradient_values):
                grad_values = evalgradients(self.subsampled_quadrature_points, gradient_values, 'matrix')
            else:
                grad_values = gradient_values
            
            p, q = grad_values.shape
            d = np.zeros((p*q,1))
            counter = 0
            for j in range(0,q):
                for i in range(0,p):
                    d[counter] = grad_values[i,j]
                    counter = counter + 1
            C = self.C_subsampled
            self.d_subsampled = d
            # Now row normalize the Cs and the ds
            if technique is None:
                raise(ValueError, 'A technique must be defined for gradient problems. Choose from stacked, equality or inequality. For more information please consult the detailed user guide.')
            else:
                if technique is 'weighted':
                    C, normalizations = rowNormalize(C)
                    d = np.dot(normalizations, d)
                    x, cond = solveCLSQ(A, b, C, d, technique)
                else:
                    x, cond = solveCLSQ(A, b, C, d, technique)
        
        return x, cond
    
    def stats(self, function_values, gradient_values=None, technique=None):
        """
        Returns statistics based on the coefficients computed. 

        :param EffectiveSubsampling object: An instance of the EffectiveSubsampling class.
        :param callable function_values: A callable function or a numpy matrix of model evaluations at the quadrature subsamples.
        :param callable gradient_values: A callable function of a numpy matrix of gradient evaluations at the quadrature subsamples.
        :param string technique: The least squares technique to be used; options include: 'weighted' (default), 'constrainedDE', 'constrainedNS'. These options only matter when using gradient evaluations. They correspond to a stacked / weighted least squares approach, a constrained approach using       direct elimination, and a constrained approach using the null space method. 
        :return: 
            * **stats_obj (Statistics)**: A Statistics object

        **Sample usage:** 
        For please see the Statistics page.
        """
        coefficients, cond = self.computeCoefficients(function_values, gradient_values, technique)
        stats_obj = Statistics(coefficients, self.index_set)
        return stats_obj
    
    def getPolynomialApproximation(self, plotting_pts, function_values, gradient_values=None, technique=None): 
        """
        Returns the polynomial approximation of a function. This routine multiplies the coefficients of a polynomial
        expansion with its corresponding basis polynomials. 
    
        :param Polynomial self: An instance of the Polynomial class
        :param: numpy array plotting_pts: The points at which the polynomial approximation should be evaluated at. When using numpy's meshgrid function to generate points, also use numpy's reshape command to ensure that the points are a n-by-d numpy matrix, where n are the number of points and d corresponds to the dimensionality of the problem.
        :param: callable function: The function that needs to be approximated (or interpolated). Call be either a callable or a numpy matrix of function values at the quadrature subsamples.
        :param: callable gradient_values: Either a callable to the gradient of the function or a numpy matrix of gradient evaluations at the quadrature subsamples.
        :param: technique: The technique used for computing gradient evaluations
        :return: polyapprox: The approximate polynomial expansion of a function
        :rtype: numpy matrix

        """
        # Check to see if we need to call the coefficients
        coefficients, cond = self.computeCoefficients(function_values, gradient_values, technique)

        stackOfParameters = self.uq_parameters
        polynomial_basis = self.index_set
        polyObject_for_basis = Polyint(stackOfParameters, polynomial_basis) 

        P , Q = polyObject_for_basis.getMultivariatePolynomial(plotting_pts)
        P = np.mat(P)
        C = np.mat(coefficients)
        polyapprox = P.T * C
        return polyapprox


################################
# Private functions!
################################
def rowNormalize(A):
    rows, cols = A.shape
    row_norms = np.mat(np.zeros((rows, 1)), dtype='float64')
    Normalization = np.mat(np.eye(rows), dtype='float64')
    for i in range(0, rows):
        temp = 0.0
        for j in range(0, cols):
            row_norms[i] = temp + A[i,j]**2
            temp = row_norms[i]
        row_norms[i] = (row_norms[i] * 1.0/np.float64(cols))**(-1)
        Normalization[i,i] = row_norms[i]
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization

# A matrix formed by a tensor grid of rows and a user-defined set of columns.
def getA(self):
    stackOfParameters = self.uq_parameters
    polynomial_basis = self.index_set
    dimensions = self.index_set.dimension

    # Crate a new PolynomialParam object to get tensor grid points & weights
    polyObject_for_pts =  Polyint(stackOfParameters)
    quadrature_pts, quadrature_wts = polyObject_for_pts.getPointsAndWeights()

    polyObject_for_basis = Polyint(stackOfParameters, polynomial_basis) 

    # Allocate memory for "unscaled points!"
    unscaled_quadrature_pts = np.zeros((len(quadrature_pts), dimensions))
    for i in range(0, dimensions):
        for j in range(0, len(quadrature_pts)):
                if (stackOfParameters[i].param_type == "Uniform"):
                    unscaled_quadrature_pts[j,i] = ((quadrature_pts[j,i] - stackOfParameters[i].lower)/(stackOfParameters[i].upper - stackOfParameters[i].lower))*2.0 - 1.0

                elif (stackOfParameters[i].param_type == "Beta" ):
                    unscaled_quadrature_pts[j,i] = (quadrature_pts[j,i] - stackOfParameters[i].lower)/(stackOfParameters[i].upper - stackOfParameters[i].lower)

    # Ensure that the quadrature weights sum up to 1.0
    quadrature_wts = quadrature_wts/np.sum(quadrature_wts)

    # Now we create another Polynomial object for the basis set!
    polynomial_expansions, no_used = polyObject_for_basis.getMultivariatePolynomial(unscaled_quadrature_pts)
    P = np.mat(polynomial_expansions)
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts

# The subsampled A matrix based on either randomized selection of rows or a QR column pivoting approach
def getSquareA(self):
    flag = self.method
    if flag == "QR" or flag is None:
        option = 1 # default option!
    elif flag == "Random":
        option = 2
    elif flag == "Convex":
        option = 3
    elif flag == "SVD-LU":
        option = 4
    else:
        raise(ValueError, "ERROR in EffectiveQuadSubsampling --> getAsubsampled(): For the third input choose from either 'QR' or 'Random'")

    A = self.A
    m , n = A.shape

    if self.no_of_evals < n :
        
        # Now if the derivative flag option is activated, we do not raise an error. Otherwise an error is raised!
        if self.uq_parameters[0].derivative_flag is None:
            raise(ValueError, "ERROR in EffectiveQuadSubsampling --> getAsubsampled(): The maximum number of evaluations must be greater or equal to the number of basis terms")

    # Now compute the rank revealing QR decomposition of A!
    if option == 1:
        Q_notused, R_notused, P = qr_MGS(A.T, pivoting=True)
        selected_quadrature_points = P[0:self.no_of_evals]
    elif option == 2:
        selected_quadrature_points = np.random.choice(m, self.no_of_evals, replace=False)
    elif option == 3:
        zhat, L, ztilde, Utilde = maxdet(A, self.no_of_evals)
        pvec = binary2indices(zhat)
        selected_quadrature_points = pvec
    elif option == 4:
        U, singular_vals, V = np.linalg.svd(A.T, full_matrices=True)
        V = np.mat(V)
        Pmat, L, U = linalg.lu( V[:, 0:self.no_of_evals])
        a, b = np.mat(Pmat).shape
        vec = np.mat( np.arange(0, a) )
        P = Pmat * vec.T
        selected_quadrature_points = []
        for i in range(0, self.no_of_evals):
            selected_quadrature_points.append(int(P[i,0]) )
        #print selected_quadrature_points

   
    # Form the "square" A matrix.
    Asquare = A[selected_quadrature_points, :]
    esq_pts = getRows(np.mat(self.tensor_quadrature_points), selected_quadrature_points)
    esq_wts = self.tensor_quadrature_weights[selected_quadrature_points]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    return Asquare, esq_pts, W, selected_quadrature_points

# Function that returns a submatrix of specific rows
def getRows(A, row_indices):
    # Determine the shape of A
    m , n = A.shape
    # Allocate space for the submatrix
    A2 = np.zeros((len(row_indices), n))
    # Now loop!
    for i in range(0, len(A2)):
        for j in range(0, n):
            A2[i,j] = A[row_indices[i], j]
    return A2

def cell2matrix(G):
    dimensions = len(G)
    G0 = G[0] # Which by default has to exist!
    C0 = G0.T
    rows, cols = C0.shape
    BigC = np.zeros((dimensions*rows, cols))
    counter = 0
    for i in range(0, dimensions):
        K = G[i].T
        for j in range(0, rows):
            for k in range(0,cols):
                BigC[counter,k] = K[j,k]
            counter = counter + 1 
    BigC = np.mat(BigC)
    return BigC

def getC(self):
    stackOfParameters = self.uq_parameters
    polynomial_basis = self.index_set
    dimensions = len(stackOfParameters)
    polyObject_for_basis = Polyint(stackOfParameters, polynomial_basis) 
    points, weights = polyObject_for_basis.getPointsAndWeights()
    not_used, C = polyObject_for_basis.getMultivariatePolynomial(points)
    return C
