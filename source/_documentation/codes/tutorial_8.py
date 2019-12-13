from equadratures import *
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    		return 1.0/(2.0 + 16*(x[0] - 0.1)**2 + 25*(x[1] + 0.1)**2 )

def tensor():
    order = 100
    x1 = Parameter(lower=-1, upper=1, order=order, distribution='Uniform')
    x2 = Parameter(lower=-1, upper=1, order=order, distribution ='Uniform')

    tensor = Basis('tensor-grid')
    myPoly = Poly([x1, x2], tensor, method='numerical-integration')
    myPoly.set_model(function)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(myPoly.get_points()[:,0], myPoly.get_points()[:,1] , marker='o', s=2, color='tomato')
    plt.xlabel('$s_1$', fontsize=13)
    plt.ylabel('$s_2$', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig('../Figures/tutorial_8_fig_a.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)

    x, y, z, max_order = vector_to_2D_grid(myPoly.get_coefficients(), myPoly.basis.get_elements() )
    G = np.log10(np.abs(z))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cax = plt.scatter(x, y, s=30, marker='o', c=G, cmap='jet', alpha=1.0, vmin=-16.0, vmax=0.)
    plt.xlim(-0.5, max_order)
    plt.ylim(-0.5, max_order)
    plt.xlabel('$i_1$', fontsize=13)
    plt.ylabel('$i_2$', fontsize=13)
    cbar = plt.colorbar(extend='neither', spacing='proportional',
                orientation='vertical', shrink=0.8, format="%.0f")
    cbar.ax.tick_params(labelsize=13)
    plt.savefig('../Figures/tutorial_8_fig_b.png',   dpi=200, bbox_inches='tight')

def sparse():
