from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma

class Test_ND_Nataf(TestCase):
    """ This class compares the mean and the variance of 
        transformed distribution, both for the direct 
        Nataf and for the inverse Nataf case.

        Expected values: 
        Results of Direct tranformation:
            mean     = 0.0
            variance = 1.0
        Results of Inverse transformation:
            mean     = the same value of input
            variance = the same value of input
    """    
    def test_mixed(self):
        """ A set of mixed distributions will be tested
            they wil be firstly mapped into a new
            standard space thanks to the direct transformation;
            then the result of the direct Nataf will be used
            into the inverse transformation and data will 
            return to their physical space.
        """
        mean1 = 0.4
        var1  = 1.3
        low1  = 0.2
        upp1  = 1.15
        
        mean2 = 0.7
        var2  = 3.0
        low2  = 0.4
        upp2  = 0.5

        D = list()
        
        D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=1.0))
	D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=4.0))
	D.append(Parameter(order=3, distribution='uniform', lower=0.05, upper=0.99))
	D.append(Parameter(order=3, distribution='uniform', lower=0.5, upper=0.8))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 1.0, shape_parameter_B=16.0))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 1.0, shape_parameter_B=16.0))
	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
	D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
	D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
	D.append(Parameter(order=3, distribution='Chebyshev', upper=1.0, lower=0.0))
	D.append(Parameter(order=3, distribution='Chebyshev', upper=0.99, lower=0.01))
	D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=14))
	D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=14))
	D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
	D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
	D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 1.7, shape_parameter_B = 0.8))
	D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 0.7, shape_parameter_B = 0.8))
	D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
	D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
	D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 100., shape_parameter_B =25.0**2, upper = 150., lower = 50.))
	D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 100., shape_parameter_B =25.0**2, upper = 150., lower = 50.))

	""" A default correlation matrix is defined in the following for statement:
			"""
	R = np.identity(len(D))
	for i in range(len(D)): 
    	    for j in range(len(D)):
        	if i==j:
            	   continue
        	else:
            	   R[i,j] = 0.60

	""" instance of Nataf class:
    	the distribution which belong to the list D will be correlated
    	with the matrix R
	"""
	obj = Nataf(D,R)

	""" Random samples are cought inside each distribution
	"""
	o = obj.getCorrelatedSamples(N=300)
	oo = obj.getUncorrelatedSamples(N=300)

	""" the following lines select the first two different
    	correlated distributions inside the set of
    	getCorrelated results.
	"""
	t = o[:,0]  # correlated 1st
	tt = o[:,1] # correlated 2nd

	print '-------------------------------------------------------------------'
	print '________test the mean and the variance after getCorrealed:_________'
	print 'mean of uncorrelated input: FROM OBJECT', obj.D[0].mean, obj.D[1].mean
	print 'mean of correlated outputs', np.mean(t) , np.mean(tt) 
	print 'variance of uncorrelated inputs: FROM OBJECT', obj.D[0].variance, obj.D[1].variance 
	print 'variance of correlated outputs', np.var(t) , np.var(tt) 
	print '-------------------------------------------------------------------'
	#------------------------------------------------------#

	""" testing transformations: direct
	"""
	u = obj.C2U(o)
	#------------------------------------------------------#
	# testing the mean and the variance of output marginals
	print '-------------------------------------------------------------------'
	print '__testing the mean and the variance after direct nataf transf._____'
	print 'direct transformation:'
	for i in range(len(D)):
    		print 'mean of ',i,'output:', np.mean(u[:,i])
	for i in range(len(D)):
    		print 'variance of ',i,'output:', np.var(u[:,i])
	print '-------------------------------------------------------------------'
	#------------------------------------------------------#
	""" testing transformations: inverse
	"""
        c = obj.U2C(u)
	#------------------------------------------------------#
	# testing the mean and the variance of output marginals
	print '-------------------------------------------------------------------'
	print '__testing the mean and the variance after inverse nataf transf.____'
	print '-----------------------------------------------'
	print 'inverse transformation:'
	for i in range(len(D)):
    		print 'mean of ',i,'th output:', np.mean(c[:,i])
	for i in range(len(D)):
    		print 'variance of ',i,'th output:', np.var(c[:,i])
	print '-----------------------------------------------'

def testbasic(self):
    print 'done!'

if __name__=='__main__':
    unittest.main()

