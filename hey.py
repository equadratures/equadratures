from equadratures.induced_sampling import InducedSampling
from equadratures.parameter import Parameter
from equadratures.basis import Basis

dimension = 3 
sampling_ratio = 10
parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
basis = Basis("total order", [3]*dimension)

induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

quadrature_points = induced_sampling.samples()

