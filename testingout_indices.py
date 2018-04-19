from equadratures import *
import numpy as np
k = 3
x1 = Parameter(distribution='Uniform', lower=-1., upper=1., order=k)
polybasis = Basis('Tensor')
G = Polylsq(parameters=[x1, x1], basis=polybasis, mesh='tensor', optimization='none', oversampling=1.)
print polybasis.elements
m, n = G.A.shape
print m, n
print np.random.choice(m, n, replace=False)
for i in range(0, m):
    print np.unravel_index(i, dims=(k+1, k+1), order='C')

