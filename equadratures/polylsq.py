"""Finding coefficients via least squares"""
from parameter import Parameter
from basis import Basis
from poly import Poly
import numpy as np
from utils import evalfunction, evalgradients, cell2matrix
from scipy.linalg import qr
from qr import solveCLSQ
import matplotlib.pyplot as plt
from convex import maxdet, binary2indices
class Polylsq(Poly):
    """
    This class defines a Polylsq (polynomial via least squares) object
    """
    def __init__(self, parameters, basis, mesh, optimization, oversampling, gradients=False):
        super(Polylsq, self).__init__(parameters, basis)
        self.mesh = mesh
        self.optimization = optimization
        self.oversampling = oversampling
        self.gradients = gradients
        n = self.basis.cardinality
        m_big = int(np.round(7 * n * np.log(n)))
        m_refined = int(np.round(self.oversampling * n))
        
        # Check that oversampling factor is greater than 1.2X of basis at the minimum
        if m_refined > m_big:
            raise(ValueError, 'Polylsq::__init__:: Oversampling factor should be greater than 1.')

        # Methods!
        if self.mesh.lower() == 'tensor':
            pts, wts_orig = super(Polylsq, self).getTensorQuadratureRule() # original weights sum up to 1
            wts = np.sqrt(wts_orig)  
        elif self.mesh.lower() == 'chebyshev':
            pts = np.cos(np.pi * np.random.rand(m_big, self.dimensions ))
            wts = float(n * 1.0)/float(m_big * 1.0) * 1.0/np.sum( (super(Polylsq, self).getPolynomial(pts))**2 , 0)
            wts_orig = wts * 1.0/np.sum(wts)
            wts = np.sqrt(wts_orig)
        elif self.mesh.lower() == 'random':
            pts = np.zeros((m_big, self.dimensions))
            for i in range(0, self.dimensions):
                univariate_samples = self.parameters[i].getSamples(m_big)
                for j in range(0, m_big):
                    pts[j, i] = univariate_samples[j]
            wts = float(n * 1.0)/float(m_big * 1.0) * 1.0/np.sum( (super(Polylsq, self).getPolynomial(pts))**2 , 0)
            wts_orig = wts * 1.0/np.sum(wts)
            wts = np.sqrt(wts_orig)
        else:
            raise(ValueError, 'Polylsq:__init___:: Unknown mesh! Choose between tensor, chebyshev, random or induced please.')

        if self.gradients is False:
            self.__gradientsFalse(pts, wts, m_refined, wts_orig)
        elif self.gradients is True:
            self.__gradientsTrue(pts, wts, m_refined, wts_orig)  
    def __gradientsTrue(self, pts, wts, m_refined, wts_orig):
        if self.optimization.lower() == 'greedy':
            P = super(Polylsq, self).getPolynomial(pts)
            W = np.mat( np.diag(wts))
            
            A = W * P.T
            __, __, pvec = qr(A.T, pivoting=True)
        else:
            raise(ValueError, 'Polylsq:__init___:: Unknown optimization technique! Choose between greedy or newton please.')
        
        index = m_refined
        rows = 1e15
        rank = 1e15
        # While loop to estimate the minimum number of rows in Az that satisfy the condition:
        # rank (Az ; Cz) >= oversampling * n
        # where n is the number of columns.
        while  (rows >= m_refined):
            z = pvec[0:index]
            refined_pts = pts[z]
            Az = A[z, :]
            Wz = np.diag(wts[z])
            Wz = 1.0/np.sum(wts[z]) * Wz
            self.Az = Az
            self.Wz = Wz
            dPcell = super(Polylsq, self).getPolynomialGradient(refined_pts)
            dP = cell2matrix(dPcell)
            M = np.vstack([Az, dP])
            rows, cols = M.shape
            rank = np.linalg.matrix_rank(M)
            del dPcell, dP 
            index = index - 1
            print 'Iterating: Current rank'+str(rank)+'  \t  Rows in A:'+str(rows)
            del M, Az  
        index = index + 1
        z = pvec[0:m_refined]
        refined_pts = pts[z]
        Pz = super(Polylsq, self).getPolynomial(refined_pts)
        wts_orig_normalized =  wts_orig[z] / np.sum(wts_orig[z])
        Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )
        self.Az = Wz * Pz.T
        self.Wz = Wz
        dPcell = super(Polylsq, self).getPolynomialGradient(refined_pts)
        dP = cell2matrix(dPcell)
        self.Cz = dP 
        self.pts = refined_pts  
    def __gradientsFalse(self, pts, wts, m_refined, wts_orig):
        P = super(Polylsq, self).getPolynomial(pts)
        W = np.mat( np.diag(wts))
        A = W * P.T
        if self.optimization.lower() == 'greedy':    
            __, __, pvec = qr(A.T, pivoting=True)
            z = pvec[0:m_refined]
        if self.optimization.lower() == 'newton':
            zhat, L, ztilde, Utilde = maxdet(A, m_refined)
            z = binary2indices(zhat)
        else:
            raise(ValueError, 'Polylsq:__init___:: Unknown optimization technique! Choose between greedy or newton please.')
        refined_pts = pts[z]
        Pz = super(Polylsq, self).getPolynomial(refined_pts)
        wts_orig_normalized =  wts_orig[z] / np.sum(wts_orig[z])
        Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )
        self.Az =  Wz * Pz.T
        self.A = A
        self.Wz = Wz
        self.pts = refined_pts
    def quadraturePoints(self):
        return self.pts
    def computeCoefficients(self, func, gradfunc=None):
        # If there are no gradients, solve via standard least squares!
        if self.gradients is False:
            p, q = self.Wz.shape
            # Get function values!
            if callable(func):
                y = evalfunction(self.pts, func)
            else:
                y = func
            self.bz = np.dot( self.Wz ,  np.reshape(y, (p,1)) )
            alpha = np.linalg.lstsq(self.Az, self.bz, rcond=None) 
            self.coefficients = alpha[0]
        # If there are gradients then use a constrained least squares approach!
        elif self.gradients is True and gradfunc is not None:
            p, q = self.Wz.shape
            # Get function values!
            if callable(func):
                y = evalfunction(self.pts, func)
            else:
                y = func
            # Get gradient values!
            if callable(func):
                grad_values = evalgradients(self.pts, gradfunc, 'matrix')
            else:
                grad_values = gradfunc
            # Assemble gradients into a single long vector called dy!     
            p, q = grad_values.shape
            d = np.zeros((p*q,1))
            counter = 0
            for j in range(0,q):
                for i in range(0,p):
                    d[counter] = grad_values[i,j]
                    counter = counter + 1
            self.dy = d
            del d, grad_values
            self.bz = np.dot( self.Wz ,  np.reshape(y, (p,1)) )
            coefficients, cond = solveCLSQ(self.Az, self.bz, self.Cz, self.dy, 'weighted')
            self.coefficients = coefficients
        elif self.gradients is True and gradfunc is None:
            raise(ValueError, 'Polylsq:computeCoefficients:: Gradient function evaluations must be provided, either a callable function or as vectors.')
        super(Polylsq, self).__setCoefficients__(self.coefficients)
    def getDesignMatrix(self):
        super(Polylsq, self).__setDesignMatrix__(self.Az)

        """
        self.method = 'QR'
        A, quadrature_pts, quadrature_wts = getA(self)
        self.A = A
        self.tensor_quadrature_points = quadrature_pts
        self.tensor_quadrature_weights = quadrature_wts
        if gradients is not None:
            self.gradients = gradients
            self.C = getC(self)
        self.no_of_basis_terms = basis.cardinality
        self.C_subsampled = None
        self.A_subsampled = None
        self.no_of_evals = None
        self.b_subsampled = None
        self.d_subsampled = None
        self.subsampled_quadrature_points = None
        self.subsampled_quadrature_weights = None # stored as a diagonal matrix??
        self.row_indices = None
        self.dimensions = len(parameters)
        self.coefficients = None  
        """
    
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
        self.basis.prune(number_of_columns_to_delete)

        # If clause for gradient case!
        if self.C is not None:
            C = self.C_subsampled
            p, q = C.shape
            C_pruned = C[0:p, 0 : (q - number_of_columns_to_delete)]
            self.C_subsampled = C_pruned
            
        self.no_of_basis_terms = self.no_of_basis_terms - number_of_columns_to_delete
    def minimumSamplesRequired(self):
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
            print self.A_subsampled
            print '******'
            print self.C_subsampled
            print rank 
            print self.no_of_basis_terms
            while rank < self.no_of_basis_terms:
                k = k + 1
                self.set_no_of_evals(k)
                rank = np.linalg.matrix_rank(np.mat( np.vstack([self.A_subsampled, self.C_subsampled]), dtype='float64') )
            return k  
    def computeCoefficients2(self, function_values, gradient_values=None, technique=None):
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
        self.coefficients = x
        return x, cond
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
def getA(self):
    quadrature_pts, quadrature_wts = super(Polylsq, self).getTensorQuadratureRule()
    P = super(Polylsq, self).getPolynomial(quadrature_pts)
    W = np.mat( np.diag(np.sqrt(quadrature_wts)))
    A = W * P.T
    return A, quadrature_pts, quadrature_wts
def getSquareA(self):
    A = self.A
    m , n = A.shape
    if self.no_of_evals < n :
        if self.gradients is None:
            raise(ValueError, "ERROR in EffectiveQuadSubsampling --> getAsubsampled(): The maximum number of evaluations must be greater or equal to the number of basis terms")

    __, __, pvec = qr(A.T, pivoting=True)
    selected_quadrature_points = self.tensor_quadrature_points[pvec[0:n],:]
    Asquare = A[pvec[0:n], :]
    esq_pts = getRows(np.mat(self.tensor_quadrature_points), pvec[0:n])
    esq_wts = self.tensor_quadrature_weights[pvec[0:n]]
    W = np.mat(np.diag(np.sqrt(esq_wts)))
    return Asquare, esq_pts, W, pvec[0:n]

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
    C = super(Polylsq, self).getPolynomialGradient(self.tensor_quadrature_points)
    return C
