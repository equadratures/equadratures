from equadratures.induced_sampling import InducedSampling
from equadratures.parameter import Parameter
from equadratures.basis import Basis

import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

dimension = 3
sampling_ratio = 10
parameters = [Parameter(3, "Uniform", upper=1, lower=-1)]*dimension
basis = Basis("total order", [3]*dimension)

induced_sampling = InducedSampling(parameters, basis, sampling_ratio, "qr")

quadrature_points, weights, polynomial = induced_sampling.samples()

ATA = np.matmul(np.matmul(polynomial, np.diag(weights)), polynomial.transpose())
print(ATA.shape)
imshow(ATA)
plt.colorbar()
plt.show()
