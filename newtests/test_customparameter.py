# Sample test utility!
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt

# Select a custom distribution!
loc, scale = 0., 1.
s = np.random.laplace(loc, scale, 5000)

# Set up the parameter!
p1 = Parameter(param_type='Custom', order=4, data=s)
x, y = p1.getPDF(50000)
yy = p1.getSamples()

# Generate the plots!
lineplot(x, y, 'Samples', 'Kernel density estimate')
histogram(yy, 'Samples', 'Histogram')