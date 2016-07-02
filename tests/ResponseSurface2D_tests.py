#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.Integrals as integrals
import effective_quadratures.Utils as utils
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

" Given an arbitrary 2D function, generate a polynomial response surface!"

def main():

    # Uq parameters setup.
    order = 7
    derivative_flag = 0 # derivative flag
    min_value = -1.5
    max_value = 1.5
    parameter_A = 0
    parameter_B = 0
    first_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    second_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters = [first_parameter, second_parameter]

    # Index set setup - don't need one for a tensor grid...but do need one for a sparse grid.
    basisObject = IndexSet("sparse grid", [], 5, "linear", 2)

    # Define a [-2,2] grid with 20 points in each direction
    num_elements = 50
    pts, x1, x2 = utils.meshgrid(-1.5, 1.5, num_elements,num_elements)

    # Setup the polyparent object
    model_approx_obj = PolyParent(uq_parameters, 'spam', basisObject)
    V, evaled_pts = PolyParent.getPolynomialApproximation(model_approx_obj, function, pts)
    print evaled_pts
    # Now we "reshape" V
    Z = np.reshape(V, (num_elements,num_elements) )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.plot_surface(ax, x1, x2, Z,cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0, alpha=0.5)
    Axes3D.scatter(ax, evaled_pts[:,0], evaled_pts[:,1], zs=0, s=50, c='r', marker='o')
    ax.set_zlim3d(0,2500)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Rosenbrock function approximated using a sparse grid')
    plt.show()

# Model or function!
def function(x):
    a = 1
    b = 100
    fun = (a - x[0])**2 + b*(x[1] - x[0]**2)**2
    return fun


main()
