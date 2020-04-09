from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
VALUE = 13
plt.rcParams.update({'font.size': VALUE})

tensor = Basis('tensor-grid', [4,4,4])
elements = tensor.elements

# Tensor grid!
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elements[:,0], elements[:,1], elements[:,2],  marker='s', s=80, color='crimson')
ax.set_xlabel('$i_i$')
ax.set_ylabel('$i_2$')
ax.set_zlabel('$i_3$')
plt.savefig('../Figures/tutorial_5_fig_a.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)

# Sparse grid
sparse = Basis('sparse-grid', level=2, growth_rule='linear')
sparse.dimensions = 3
a, b, c = sparse.get_basis()
print(a)


# Euclidean grid!
euclid = Basis('euclidean-degree', [4,4,4])
elements = euclid.elements
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elements[:,0], elements[:,1], elements[:,2],  marker='s', s=80, color='crimson')
ax.set_xlabel('$i_i$')
ax.set_ylabel('$i_2$')
ax.set_zlabel('$i_3$')
plt.savefig('../Figures/tutorial_5_fig_b.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)

# Total order grid!
total = Basis('total-order', [4,4,4])
elements = total.elements
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elements[:,0], elements[:,1], elements[:,2],  marker='s', s=80, color='crimson')
ax.set_xlabel('$i_i$')
ax.set_ylabel('$i_2$')
ax.set_zlabel('$i_3$')
plt.savefig('../Figures/tutorial_5_fig_c.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)

# Hyperbolic basis!
hyper = Basis('hyperbolic-basis', [4,4,4], q=0.5)
elements = hyper.elements
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elements[:,0], elements[:,1], elements[:,2],  marker='s', s=80, color='crimson')
ax.set_xlabel('$i_i$')
ax.set_ylabel('$i_2$')
ax.set_zlabel('$i_3$')
plt.savefig('../Figures/tutorial_5_fig_d.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)