#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
import effective_quadratures.integrals as integrals
from effective_quadratures.utils import error_function
import numpy as np
# Vegetation study example Test
def main():

    # 4D function for testing!
    def fun(x):
        return np.cos(x[0]) + x[1]**2 + x[2]*x[3]
    

    #-----------------------------------------------------------------
    # Test 1: Integral with uniformly distributed parameters
    #-----------------------------------------------------------------
    x1 = Parameter(lower=-0.5, upper=0.5, points=4)
    x2 = Parameter(lower=-1, upper=2, points=4)
    x3 = Parameter(lower=-3, upper=2, points=4)
    x4 = Parameter(lower=-2, upper=1, points=4)
    parameters = [x1, x2, x3, x4]
    result, points = integrals.tensorgrid(parameters, fun)
    print result
    if np.linalg.norm(result - 99.39829845, 2) < 1e-9:
        print np.linalg.norm(result - 99.39829845, 2)
        print 'Success!'
    else:
        error_function('ERROR: Uniform integration routine not working!')    

    #-----------------------------------------------------------------
    # Test 2: Integral with normally distributed parameters
    #-----------------------------------------------------------------
    x1 = Parameter(param_type = 'Gaussian', shape_parameter_A=0, shape_parameter_B=1, points=4)
    x2 = Parameter(param_type = 'Gaussian',shape_parameter_A=0, shape_parameter_B=1, points=4)
    x3 = Parameter(param_type = 'Gaussian',shape_parameter_A=0, shape_parameter_B=1, points=4)
    x4 = Parameter(param_type = 'Gaussian',shape_parameter_A=0, shape_parameter_B=1, points=4)
    parameters = [x1, x2, x3, x4]
    result, points = integrals.tensorgrid(parameters, fun)
    print result
    if np.linalg.norm(result - 1.606, 2) < 1e-4:
        print np.linalg.norm(result - 1.606, 2)
        print 'Success!'
    else:
        error_function('ERROR: Normal integration routine not working!')     

    # Test to see if we get points and weights when function is not callable!
    points, weights = integrals.tensorgrid(parameters) 

    #-----------------------------------------------------------------
    # Test 3: Sparse grid integration rule test
    #-----------------------------------------------------------------
    x1 = Parameter(lower=-0.5, upper=0.5, points=4)
    x2 = Parameter(lower=-1, upper=2, points=4)
    x3 = Parameter(lower=-3, upper=2, points=4)
    x4 = Parameter(lower=-2, upper=1, points=4)
    parameters = [x1, x2, x3, x4]
    result, points = integrals.sparsegrid(parameters, level=5, growth_rule='exponential', function=fun)
    print result
    if np.linalg.norm(result - 99.3982, 2) < 1e-4:
        print np.linalg.norm(result - 99.3982, 2)
        print 'Success!'
    else:
        error_function('ERROR: Sparse grid integration routine not working!') 
    
    #-----------------------------------------------------------------
    # Test 4: Effective-Quadratures integration rule
    #-----------------------------------------------------------------
    x1 = Parameter(lower=-0.5, upper=0.5, points=4)
    x2 = Parameter(lower=-1, upper=2, points=4)
    x3 = Parameter(lower=-3, upper=2, points=4)
    x4 = Parameter(lower=-2, upper=1, points=4)
    parameters = [x1, x2, x3, x4]
    result, points = integrals.effectivequadratures(parameters, q_parameter=0.8, function=fun)
    print result
    if np.abs(result- 99.3982) < 1e-4:
        print np.abs(result- 99.3982)
        print 'Success!'
    else:
        error_function('ERROR: Effective-Quadratures integration routine not working!') 

main()
