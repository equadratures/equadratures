"""Sampling strategy."""
import numpy as np
class Sampling(object):
    """
    The class defines a Sampling object. It serves as a template for all sampling methodologies.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis, points=None, weights=None ):
        self.parameters = parameters
        self.basis = basis
        self.points = points
        self.weights = weights
        if self.weights is None:
            self.__set_weights()
    def __set_weights(self):
        P = self.__get_multivariate_orthogonal_polynomial(self.points)
        wts =  1.0/np.sum( P**2 , 0)
        self.weights = wts * 1.0/np.sum(wts)
    def get_points(self):
        """
        Returns the quadrature points.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.points
    def get_weights(self):
        """
        Returns the quadrature weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.weights
    def get_points_and_weights(self):
        """
        Returns the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.points, self.weights
    def __get_multivariate_orthogonal_polynomial(self):
        """
        Utility for evaluating a multivariate orthogonal polynomial at given points.

        :param Sampling self:
                An instance of the sampling class.
        """
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        p = {}

        if self.points.ndim == 1:
            no_of_points = 1
        else:
            no_of_points, __ = self.points.shape

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ , _ =  self.parameters[0]._get_orthogonal_polynomial(self.points, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stack_of_points.shape) == 1:
                    stack_of_points = np.array([stack_of_points])
                p[i] , _ , _ = self.parameters[i]._get_orthogonal_polynomial(self.points[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        polynomial = np.ones((basis_entries, no_of_points))
        for k in range(dimensions):
            basis_entries_this_dim = basis[:, k].astype(int)
            polynomial *= p[k][basis_entries_this_dim]
        return polynomial