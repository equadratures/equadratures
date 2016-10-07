#!/usr/bin/env python
"""Operations involving multivariate polynomials"""
from parameter import Parameter
from indexset import IndexSet
import numpy as np
from utils import error_function, evalfunction, find_repeated_elements, meshgrid

class Polynomial(object):
    """
    This class defines a polynomial and its associated functions. 

    :param array of Parameters uq_parameters: A list of Parameters
    :param IndexSet index_set: An instance of the IndexSet class, in case user wants to overwrite the indices
        that are obtained using the orders of the univariate parameters in Parameters uq_parameters
    
    **Sample declarations** 
    ::
        >> s = Parameter(lower=-2, upper=2, param_type='Uniform', points=4)
        >> T = IndexSet('Total order', [3,3])
        >> polyObject = Polynomial([s,s],T) # basis is defined by T

        >> s = Parameter(lower=-2, upper=2, param_type='Uniform')
        >> T = IndexSet('Tensor grid', [3,3])
        >> polyObject = Polynomial([s,s]) # Tensor basis is used
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
    
    def getIndexSet(self):
        """
        Returns orthogonal polynomials & its derivatives, evaluated at a set of points.

        :param Parameter self: An instance of the Parameter class
        :param ndarray points: Points at which the orthogonal polynomial (and its derivatives) should be evaluated at
        :param int order: This value of order overwrites the order defined for the constructor.
        :return: orthopoly, order-by-k matrix where order defines the number of orthogonal polynomials that will be evaluated
            and k defines the points at which these points should be evaluated at.
        :rtype: ndarray
        :return: derivative_orthopoly, order-by-k matrix where order defines the number of derivative of the orthogonal polynomials that will be evaluated
            and k defines the points at which these points should be evaluated at.
        :rtype: ndarray

        **Sample declaration**
        :: 
            >> x = np.linspace(-1,1,10)
            >> var6 = Parameter(points=10, param_type='Uniform', lower=-1, upper=1)
            >> poly = var6.getOrthoPoly(x)
        """
        return self.index_sets.getIndexSet()

    # Do we really need additional_orders?
    def getPointsAndWeights(self):
    
        # Initialize some temporary variables
        stackOfParameters = self.uq_parameters
        dimensions = int(len(stackOfParameters))
        
        orders = []
        for i in range(0, dimensions):
            orders.append(stackOfParameters[i].order)
        
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
        empty = np.mat([0])
        stackOfParameters = self.uq_parameters
        isets = self.index_sets
        index_set = isets.getIndexSet()
        dimensions = len(stackOfParameters)
        p = {}
        d = {}
        C_all = {}

        # Save time by returning if univariate!
        if dimensions == 1 and stackOfParameters[0].derivative_flag == 0:
            poly , derivatives =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
            derivatives = empty
            return poly, derivatives
        elif dimensions == 1 and stackOfParameters[0].derivative_flag == 1:
            poly , derivatives =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
            return poly, derivatives
        else:
            for i in range(0, dimensions):
                p[i] , d[i] = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )

        # Now we multiply components according to the index set
        no_of_points = len(stackOfPoints)
        polynomial = np.zeros((len(index_set), no_of_points))
        derivatives = np.zeros((len(index_set), no_of_points, dimensions))

        # One loop for polynomials
        for i in range(0, len(index_set)):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                polynomial[i,:] = p[k][int(index_set[i,k])] * temp
                temp = polynomial[i,:]
        
        # Second loop for derivatives!
        if stackOfParameters[0].derivative_flag == 1:
            P_others = np.zeros((len(index_set), no_of_points))

            # Going into for loop!
            for j in range(0, dimensions):
                # Now what are the remaining dimnensions?
                C_local = np.zeros((len(index_set), no_of_points))
                remaining_dimensions = np.arange(0, dimensions)
                remaining_dimensions = np.delete(remaining_dimensions, j)
                total_elements = remaining_dimensions.__len__

                # Now we compute the "C" matrix
                for i in range(0, len(index_set)): 
                    # Temporary variable!
                    P_others = np.zeros((len(index_set), no_of_points))
                    temp = np.ones((1, no_of_points))

                    # Multiply ortho-poly components in these "remaining" dimensions   
                    for k in range(0, len(remaining_dimensions)):
                        entry = remaining_dimensions[k]
                        P_others[i,:] = p[entry][int(index_set[i, entry])] * temp
                        temp = P_others[i,:]
                        if len(remaining_dimensions) == 0: # in which case it is emtpy!
                            C_all[i,:] = d[j][int(index_set[i,j])]
                        else:
                            C_local[i,:] = d[j][int(index_set[i, j])] * P_others[i,:]       
                C_all[j] = C_local
                del C_local
            return polynomial, C_all
        empty = np.mat([0])
        return polynomial, empty