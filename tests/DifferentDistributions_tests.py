#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.ComputeStats as stats
import numpy as np
import matplotlib.pyplot as plt
"""

    Test.

    Copyright (c) 2016 by Pranay Seshadri
"""
# Simple analytical function

def rosenbrock_fun(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def main():
    no_pts_x1 = 5
    no_pts_x2 = 5
    #k = 5
    #theta = 1
    mu = 1
    sigma = 2
    variance = sigma**2
    #a = 1
    #b = 5
    lambda_value = 1.5
    method = "tensor grid"
    #x1 = PolynomialParam("Gaussian", [], [], mu, variance, [], no_pts_x1)
    #x2 = PolynomialParam("Gaussian", [], [], mu, variance, [], no_pts_x2)
    #x1 = PolynomialParam("Weibull", [], [], a, b, [], no_pts_x1)
    #x2 = PolynomialParam("Weibull", [], [], a, b, [], no_pts_x2)
    #x1 = PolynomialParam("Exponential", [], [], lambda_value, [], [], no_pts_x1)
    #x2 = PolynomialParam("Exponential", [], [], lambda_value, [], [], no_pts_x2)
    x1 = PolynomialParam("TruncatedGaussian", -15, 15, mu, variance, [], no_pts_x1)
    x2 = PolynomialParam("TruncatedGaussian", -15, 15, mu, variance, [], no_pts_x2)
    #x1 = PolynomialParam("Gamma", [], [], k, theta, [], no_pts_x1)
    #x2 = PolynomialParam("Gamma", [], [], k, theta, [], no_pts_x2)
    x1x2 = []
    x1x2.append(x1)
    x1x2.append(x2)

    # spam.
    """
    method = "spam"
    growth_rule = "linear"
    level = 4
    dimension = 2
    basis = IndexSet("sparse grid", [], level, growth_rule, dimension)
    uqProblem = PolyParent(x1x2, method, basis)
    pts, wts = PolyParent.getPointsAndWeights(uqProblem)
    x, i, f = PolyParent.getCoefficients(uqProblem, rosenbrock_fun)
    mean, variance = stats.compute_mean_variance(x, i)
    print mean, variance
    """
    # Tensor grid
    method = "tensor grid"
    uqProblemT = PolyParent(x1x2, method)
    pts2, wts = PolyParent.getPointsAndWeights(uqProblemT)
    x, i, f = PolyParent.getCoefficients(uqProblemT, rosenbrock_fun)
    mean, variance = stats.compute_mean_variance(x, i)
    print mean, variance

    print pts2, wts
    #plt.scatter(pts[:,0], pts[:,1], s=70, c='b', marker='o')
    #plt.scatter(pts2[:,0], pts2[:,1], s=70, c='r', marker='o')
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    #plt.show()

main()
