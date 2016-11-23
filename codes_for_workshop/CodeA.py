#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
import numpy as np

# Setting up the parameter
s = Parameter(param_type='Beta', lower=-2, upper=5, shape_parameter_A=3, shape_parameter_B=2, points=5)
s.getPDF(300, graph=1)

# Computing 1D quadrature points and weights
points, weights = s.getLocalQuadrature()
print points, weights

# Getting the Jacobi matrix
print s.getJacobiMatrix()

# Getting the first 5 orthogonal polynomial evaluated at some points x 
x = np.linspace(-2, 5, 10)
print s.getOrthoPoly(x)