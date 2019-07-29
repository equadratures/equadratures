"""The polynomial parent class."""
from equadratures.parameter import Parameter
from equadratures.basis import Basis
import numpy as np
class Samples(Poly):
    """
    Numerical quadrature techniques.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    :param tuple samples: With the format (str, dict). Here the str argument to this input specifies the sampling strategy. Avaliable options
        are: ``monte-carlo``, ``latin-hypercube``, ``induced-sampling``, ``christoffel-sampling``, ``sparse-grid``, ``tensor-grid`` or ``user-defined``.
        The second argument to this input is a dictionary, that naturally depends on the chosen string.
        Note that ``monte-carlo``, ``latin-hypercube``, ``induced-sampling`` and ``christoffel-sampling`` are random
        sampling techniques and thus their output will vary with each instance; initialization of a random seed
        is recommended to facilitate reproducibility. The second argument to this input is a dict with the following key value structure.

        :param dict args: For ``monte-carlo``, ``latin-hypercube``, ``induced-sampling`` and ``christoffel-sampling``, the following structure
            ``{'sampling-rato': (double), 'subsampling-option': (str), 'correlation': (numpy.ndarray)}`` should be adopted. The ``sampling-ratio``
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

    **References**
        1. Seshadri, P., Iaccarino, G., Ghisu, T., (2018) Quadrature Strategies for Constructing Polynomial Approximations. Uncertainty Modeling for Engineering Applications. Springer, Cham, 2019. 1-25. `Preprint <https://arxiv.org/pdf/1805.07296.pdf>`__
        2. Seshadri, P., Narayan, A., Sankaran M., (2017) Effectively Subsampled Quadratures for Least Squares Polynomial Approximations. SIAM/ASA Journal on Uncertainty Quantification 5.1 :1003-1023. `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`__
        3. Bos, L., De Marchi, S., Sommariva, A., Vianello, M., (2010) Computing Multivariate Fekete and Leja points by Numerical Linear Algebra. SIAM Journal on Numerical Analysis, 48(5). `Paper <https://epubs.siam.org/doi/abs/10.1137/090779024>`__
        4. Joshi, S., Boyd, S., (2009) Sensor Selection via Convex Optimization. IEEE Transactions on Signal Processing, 57(2). `Paper <https://ieeexplore.ieee.org/document/4663892>`__
    """
    def __init__(self, parameters, basis, method, samples):
        super(Poly, self).__init__(parameters, basis, method, samples)

    def getTensorQuadratureRule(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.
        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.basis.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._get_local_quadrature(orders[u])
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        points = pp[:,1::]
        weights = ww

        # Return tensor grid quad-points and weights
        return points, weights
    """
    def get_tensor_grid_quadrature_rule(self):

    def generate_quadrature_rule(self):


        points_subsampled = pts[z,:]
            #quadraturePoints_subsampled = pts[z,:]
            wts_orig_normalized =  wts[z] / np.sum(wts[z]) # if we pick a subset of the weights, they should add up to 1.!
            self.A = A
        Pz = super(Polylsq, self).getPolynomial(points_subsampled)
        Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )

        wts =  1.0/(np.sum( super(Polylsq, self).getPolynomial(pts)**2 , 0) )**2
            wts = wts * 1.0/np.sum(wts)


        self.Quadrature = Quadrature(method, samples)
        self.quadrature_points = self.Quadrature.get_samples
        self.quadrature_weights = self.Quadrature.get_weights
    """
