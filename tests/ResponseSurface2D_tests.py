#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.Integrals as integrals
import effective_quadratures.Utils as utils
import matplotlib.pyplot as plt
import numpy as np

" Given an arbitrary 2D function, generate a polynomial response surface!"

def main():

    # Uq parameters setup.
    order = 5
    derivative_flag = 0 # derivative flag
    min_value = -3
    max_value = 3
    parameter_A = 0
    parameter_B = 0
    first_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    second_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters = [first_parameter, second_parameter]

    # Index set setup - don't need one for a tensor grid...but do need one for a sparse grid.
    tensorgridObject = IndexSet("tensor grid", [order, order])

    # Define a [-2,2] grid with 20 points in each direction
    pts = utils.meshgrid(-2, 2, 20,20)

    # Setup the polyparent object
    model_approx_obj = PolyParent(uq_parameters, tensorgridObject)
    V = PolyParent.getPolynomialApproximation(model_approx_obj, pts)







# Model or function!
def function(x):
    a = 1
    b = 100
    fun = (a - x[0])**2 + b*(x[1] - x[0]**2)**2
    return fun
