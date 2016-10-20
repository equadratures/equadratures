#!/usr/bin/env python
from effective_quadratures.indexset import IndexSet
import numpy as np
import matplotlib.pyplot as plt

# Setting up the parameter
#tensor = IndexSet('Tensor grid', [3, 3])
total = IndexSet('Total order', [3, 3])
I = total.getIndexSet()
print I

# Lets plot the elements
plt.scatter(I[:,0], I[:,1], s=90, c='r', marker='o')
plt.xlabel('i1')
plt.ylabel('i2')
plt.show()