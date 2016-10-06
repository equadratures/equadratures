#!/usr/bin/env python
"""Operations involving multivariate polynomials"""
from parameter import Parameter
from indexset import IndexSet
import numpy as np
from utils import error_function, evalfunction, find_repeated_elements

class Polynomial(object):
    """
    This class defines a polynomial and its associated functions. 

    :param array of Parameters uq_parameters: A list of Parameters
    :param IndexSet index_set: An instance of the IndexSet class, in case user wants to overwrite the indices
        that are obtained using the orders of the univariate parameters in Parameters uq_parameters
    
    **Sample declarations** 
    ::
        >> s = Parameter(lower=-2, upper=2, param_type='Uniform')
        >> T = IndexSet('Tensor grid', [3,3])
        >> Polynomial([s,s],T)
    """

    # Constructor
    def __init__(self, uq_parameters, index_sets=None):
    
        self.uq_parameters = uq_parameters

        # Here we set the index sets if they are not provided
        if index_sets is None:
            # Determine the highest orders for a tensor grid
            highest_orders = []
            for i in range(0, len(uq_parameters)):
                highest_orders.append(uq_parameters[i].order)
            
            self.index_sets = IndexSet('Tensor grid', highest_orders)
        else:
            self.index_sets = index_sets

    def getPointsAndWeights(self, additional_orders=None):
    
        # Initialize some temporary variables
        dimensions = int(len(stackOfParameters))
        orders = []

        # Check for extra input argument!
        if additional_orders is None:
            for i in range(0, dimensions):
                orders.append(stackOfParameters[i].order)
        else:
            for i in range(0, dimensions):
                orders.append(additional_orders[i])

        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        # number of parameters
        # For loop across each dimension
        for u in range(0,dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = stackOfParameters[u].getLocalQuadrature(orders[u])

            # Tensor product of the weights
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        points = pp[:,1::]
        weights = ww

        # Now re-scale the points and return only if its not a Gaussian!
        for i in range(0, dimensions):
            for j in range(0, len(points)):
                if (stackOfParameters[i].param_type == "Uniform"):
                    #points[j,i] = points[j,i] * ( stackOfParameters[i].upper_bound - stackOfParameters[i].lower_bound) + stackOfParameters[i].lower_bound
                    points[j,i] = 0.5 * ( points[j,i] + 1.0 )*( stackOfParameters[i].upper - stackOfParameters[i].lower) + stackOfParameters[i].lower

                elif (stackOfParameters[i].param_type == "Beta" ):
                    points[j,i] =  ( points[j,i] )*( stackOfParameters[i].upper - stackOfParameters[i].lower) + stackOfParameters[i].lower

                elif (stackOfParameters[i].param_type == "Gaussian"):
                    points[j,i] = points[j,i] # No scaling!

    # Return tensor grid quad-points and weights
    return points, weights
    
    def getMultivariatePolynomial(self, stackOfPoints):

        # "Unpack" parameters from "self"
        stackOfParameters = self.uq_parameters
        isets = self.index_sets
        index_set = isets.getIndexSet()
        dimensions = len(stackOfParameters)
        p = {}
        d = {}

        # Save time by returning if univariate!
        if(dimensions == 1):
            poly , derivatives =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
            return poly, derivatives
        else:
            for i in range(0, dimensions):
                p[i] = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )
                #print G
                #p[i] = G
                #d[i] = D
        # Now we multiply components according to the index set
        no_of_points = len(stackOfPoints)
        polynomial = np.zeros((len(index_set), no_of_points))
        derivatives = np.zeros((len(index_set), no_of_points, dimensions))

        # One loop for polynomials
        for i in range(0, len(index_set)):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                polynomial[i,:] = p[k][0][int(index_set[i,k])] * temp
                temp = polynomial[i,:]
                #print polynomial
        
        # Second loop for derivatives!
        if stackOfParameters[0].derivative_flag == 1:
            print 'WIP'

        return polynomial, derivatives
