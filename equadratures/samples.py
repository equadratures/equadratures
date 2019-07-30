"""The Sample template."""
from equadratures.parameter import Parameter
from equadratures.basis import Basis
import numpy as np
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
            self.

    def get_points_and_weights(self):
        return self.mysamples.get_points_and_weights()
    def get_points(self):
        return self.mysamples.get_points()
    def set_model(self, fun):
        self.mysamples.set_model(fun)
    def get_coefficients(self):
        self.mysamples.get_coefficients()