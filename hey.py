from equadratures.sampling_methods.induced import Induced
from equadratures.parameter import Parameter
from equadratures.basis import Basis

import time

start_time = time.time()
dimension = 1
parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
basis = Basis("total-order")

induced_sampling = Induced(parameters, basis)
print(f"time taken:{time.time()-start_time}")
quadrature_points = induced_sampling.get_points()
quadrature_weights = induced_sampling.get_weights()
print(quadrature_points)
