from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.stats import Statistics
from equadratures.basis import Basis
from equadratures.correlations import Correlations
from equadratures.optimisation import Optimisation
from equadratures.subspaces import get_active_subspace, variable_projection
from equadratures.poly import evaluate_model, evaluate_model_gradients, vector_to_2D_grid
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
VERSION_NUMBER = 8.0
