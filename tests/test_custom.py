from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt

class Test_Custom(TestCase):
    """ This class compares the probabilitity density functions,
        the cumulative distribution functions, the inverse of 
        the cdf obtained using the related methods of the Custom
        class, given a set of points which belong to a known
        distribution.
    """
    def testValues(self): 
        D = list() # list of object that will be tested
        D.append(Parameter(order = 3 , distribution='gaussian', shape_parameter_A = 0.0, shape_parameter_B=1.0))
        D.append(Parameter(order=5, distribution='uniform', lower=-1., upper =1.))
        D.append(Parameter(order=5, distribution='rayleigh', shape_parameter_A =1.))
        D.append(Parameter(order=5, distribution='beta', shape_parameter_A = 1., shape_parameter_B = 1., lower=0., upper = 1.))
        D.append(Parameter(order=5, distribution='Chebyshev', upper = 1., lower=0.))
        D.append(Parameter(order=5, distribution='Chisquared', shape_parameter_A = 14))
        D.append(Parameter(order=5, distribution='exponential', shape_parameter_A = 0.7))
        D.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A = 1., shape_parameter_B = 1., lower = 0.5, upper = 1.5))

        for i in range(len(D)):
            data1 = D[i].getSamples(m=500) 
            data1 = np.sort(data1)
            custom = Parameter(order=3, distribution='custom', data=data1)

            # pdf :

            pdf  = np.array(custom.getPDF(data1))
            g_pdf = np.array(D[i].getPDF(data1))

            plt.figure()
            plt.grid(linewidth = 0.4, color = 'k')
            plt.plot(data1, pdf, 'bo', label ='pdf of Custom')
            plt.plot(data1, g_pdf, 'ro', label=D[i].name)
            plt.legend(loc='lower center')
            plt.show()

        for i in range(len(D)):
            data1 = np.array(D[i].getSamples(m=500))
            data1 = np.reshape(data1, (500,1)) 
            custom = Parameter(order=3, distribution='custom', data=data1)
            #cdf  = np.array(custom.getCDF(data1))
            #g_cdf = np.array(D[i].getCDF(data1))

            #plt.figure()
            #plt.grid(linewidth = 0.4, color = 'k')
            #plt.plot(data1, cdf, 'bo', label='cdf of Custom Class')
            #plt.plot(data1, g_cdf, 'ro', label= D[i].name)
            #plt.legend(loc='upper left')
            #plt.show()

        for i in range(len(D)):
            data1 = D[i].getSamples(m=500) 
            custom = Parameter(order=3, distribution='custom', data=data1)
        
            #icdf = custom.getiCDF(cdf)
            #g_icdf = D[i].getiCDF(g_cdf)
        
            #plt.figure()
            #plt.grid(linewidth = 0.4, color = 'k')
            #plt.plot(cdf, icdf, 'bo', label='icdf of Custom Class')
            #plt.plot(g_cdf, g_icdf, 'ro', label= D[i].name)
            #plt.legend(loc='upper left')
            #plt.show()
        

    def testCustomNataf(self):
        """ this method tests the Nataf transformation in 2 dimensions using the custom class
            defined by the user.
        """
        print 'second method commented'
     
        source1 = Parameter(distribution='uniform', order=5, lower=0.0, upper =1.0)
        source2 = Parameter(distribution='gaussian', order=5, shape_parameter_A = 10., shape_parameter_B=1.)
        
        pt1 = source1.getSamples(1000)
        pt2 = source2.getSamples(1000)

        D = list()
        D.append(Parameter(distribution='custom', data=pt1, order=5))
        D.append(Parameter(distribution='custom', data=pt2, order=5))
        R = np.matrix([[1., 0.7], [0.7, 1.]])

        obj = Nataf(D, R)
        
        corr = obj.getCorrelatedSamples(N=400)
        direct_transf  = obj.C2U(corr)
        #print 'direct transformation:'
        #print direct_transf
        inverse_transf = obj.U2C(direct_transf)
        #print 'inverse transformation:'
        #print inverse_transf

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(corr[:,0], corr[:,1], 'ro', label='original correlated')
        plt.plot(direct_transf[:,0], direct_transf[:,1], 'bo', label='direct transf')
        plt.legend(loc='upper left')
        plt.title('Direct transformations')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(inverse_transf[:,0], inverse_transf[:,1], 'ro', label='inverse transf')
        plt.plot(corr[:,0], corr[:,1], 'bx', label='original correlated')
        plt.legend(loc='upper left')
        plt.title('Inverse transformations')
        plt.show()

        print '-----------------------------------------------------------------------'
        print '______________________ DIRECT TRANSFORMATION___________________________'
        print 'Mean of Correlated data:', np.mean(corr[:,0]), 'and', np.mean(corr[:,1])
        print 'Mean of Transformed data:', np.mean(direct_transf[:,0]), 'and', np.mean(direct_transf[:,1])
        print 'Variance of Correlated data:', np.var(corr[:,0]),'and', np.var(corr[:,1])
        print 'Variance of Transformed data:', np.var(direct_transf[:,0]), 'and', np.var(direct_transf[:,1])
        print '-----------------------------------------------------------------------'

        print '-----------------------------------------------------------------------'
        print '______________________ INVERSE TRANSFORMATION__________________________'
        print 'Mean of Correlated data:', np.mean(corr[:,0]), 'and', np.mean(corr[:,1])
        print 'Mean of Transformed data:', np.mean(inverse_transf[:,0]), 'and', np.mean(inverse_transf[:,1])
        print 'Variance of Correlated data:', np.var(corr[:,0]),'and', np.var(corr[:,1])
        print 'Variance of Transformed data:', np.var(inverse_transf[:,0]), 'and', np.var(inverse_transf[:,1])
        print '-----------------------------------------------------------------------'
        
    def testNdimensions(self):
        """ this method tests the Nataf transformation with N dimensions;
            each marginal is an object of custom class and receives different
            kind of data for the instance.
        """
        D = list()
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))  
        D.append(Parameter(distribution='uniform', order=5, lower=-1., upper =2.))
        
        L= len(D)
        pnts = list() 
        M = list()
        R = np.zeros((len(D), len(D)))
        for i in range(len(D)):
                pnts.append(D[i].getSamples(500))
                M.append(Parameter(distribution='custom', order=5, data=pnts[i]))
        
        for i in range(L):
            for j in range(L):
                if i==j:
                    R[j,i] = 1.
                else: 
                    R[j,i] = 0.7
        obj = Nataf(M,R)
        corr_samples = obj.getCorrelatedSamples(400)
        direct_trans = obj.C2U(corr_samples)
        inverse_tran = obj.U2C(direct_trans)

        print '-------------------------------------------------'
        print '______________DIRECT TRANSFORMATION______________'
        print 'mean of transformed marginals:'
        for i in range(len(D)):
            print 'Marginal',i, 'has mean=', np.mean(direct_trans[:,i])
        print 'variance of transformed marginals:'
        for i in range(len(D)):
            print 'Marginal',i, 'has variance=', np.var(direct_trans[:,i])
        print '-------------------------------------------------'
        print '______________INVERSE TRANSFORMATION_____________' 
        print 'LIST OF MEAN VALUES: '
        for i in range(len(D)):    
            print 'original',i, 'has',np.mean(corr_samples[:,i]),'inverse transf.',i, 'has',np.mean(inverse_tran[:,i])
        print 'LIST OF VARIANCE VALUES:'
        for i in range(len(D)):
            print 'original',i, 'has',np.var(corr_samples[:,i]),'inverse transf,',i,'has', np.var(inverse_tran[:,i])
        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(direct_trans[:,1], direct_trans[:,2], 'bo', label='1VS2')
        plt.plot(corr_samples[:,1], corr_samples[:,2], 'ro', label='original')
        plt.legend(loc='upper left')
        plt.axis('equal')
        plt.title('first VS second transformed')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(direct_trans[:,0], direct_trans[:,1], 'bo', label='0VS1')
        plt.plot(corr_samples[:,0], corr_samples[:,1], 'ro', label='original')
        plt.legend(loc='upper left')
        plt.axis('equal')
        plt.title('0 VS first transformed')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(direct_trans[:,0], direct_trans[:,2], 'bo', label='0VS2')
        plt.plot(corr_samples[:,0], corr_samples[:,2], 'ro', label='original')
        plt.legend(loc='upper left')
        plt.axis('equal')
        plt.title('0 VS second transformed')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(corr_samples[:,0], corr_samples[:,1], 'bo', label='original')
        plt.plot(inverse_tran[:,0], inverse_tran[:,1], 'ro', label='inverse')
        plt.legend(loc='upper left')
        plt.title('inverse VS original: 0 VS 1')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(corr_samples[:,0], corr_samples[:,2], 'bo', label='original')
        plt.plot(inverse_tran[:,0], inverse_tran[:,2], 'ro', label='inverse')
        plt.legend(loc='upper left')
        plt.title('inverse VS original: 0 VS 2')
        plt.show()

        plt.figure()
        plt.grid(linewidth=0.5, color='k')
        plt.plot(corr_samples[:,1], corr_samples[:,2], 'bo', label='original')
        plt.plot(inverse_tran[:,1], inverse_tran[:,2], 'ro', label='inverse')
        plt.legend(loc='upper left')
        plt.title('inverse VS original: 1 VS 2')
        plt.show()


if __name__== '__main__':
    unittest.main()
  
