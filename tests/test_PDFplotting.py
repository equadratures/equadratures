#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
import numpy as np

# A few utils for testing some of the new plotting functionalities in Effective-Quadratures!
# 1. Check PDFs of parameters using the iCDF function calls
# 2. Check PDFs of polynomial objects!
X = Parameter(points=3, shape_parameter_A=0, shape_parameter_B=2.5, param_type='Gaussian')
X.getSamples(1000, graph=1)