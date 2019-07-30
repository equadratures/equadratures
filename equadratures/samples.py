"""The Sample template."""
from equadratures.parameter import Parameter
from equadratures.basis import Basis
from equadratures.quadrature import sparse_grid, tensor_grid, tensor_grid_subsampled
import numpy as np

class Userdefined(object):
    def __init__(self, parameters, basis, inputs outputs, coefficient_computation):
    """
    The user defined case.
    """
    def __init(self, parameters, basis, inputs, outputs, solver_function):
        self.X = inputs
        if len(self.X.shape) == 1:
            self.X = np.reshape(self.X,(len(self.X),1))
        assert self.X.shape[1] == len(self.parameters) # Check that x is in the correct shape
        self.y = training_outputs
        assert self.y.shape[0] == self.X.shape[0]
        self.__set_points_and_weights()
        self.__get_design_matrix()

    def __set_points_and_weights(self, ):

    def get_points_and_weights(self):
        return

class Samples(object):
    """
    Computes samples.
    """
    def __init__(self, parameters, basis, mesh=None, subsampling_algorithm=None, subsampling_ratio=None, inputs=None, outputs=None, level=None, growth_rule=None, solver_function=None):
        self.parameters = parameters
        self.dimensions = len(self.parameters)
        self.basis = basis
        self.mesh = mesh
        self.subsampling_algorithm = subsampling_algorithm
        self.subsampling_ratio = subsampling_ratio
        self.inputs = inputs
        self.outputs = outputs
        self.solver_function = solver_function

        # Setup code to compute the points!
        if self.mesh == 'sparse-grid':
            self.mysample = sparse_grid(parameters=self.parameters, basis=self.basis, level=self.level, dimensions=self.dimensions, growth_rule=self.growth_rule)
        elif self.mesh == 'tensor-grid' and self.subsampling_algorithm == None:
            self.mysample = tensor_grid(parameters=self.parameters, basis=self.basis)
        elif self.mesh == 'tensor-grid' and self.subsampling_algorithm is not None:
            self.mysample = tensor_grid_subsampled(parameters=self.parameters, basis=self.basis, solver_function=self.solver_function)


    def get_points_and_weights(self):
        return self.mysamples.get_points_and_weights()
    def get_points(self):
        return self.mysamples.get_points()
    def set_model(self, fun):
        self.mysamples.set_model(fun)
    def get_coefficients(self):
        self.mysamples.get_coefficients()