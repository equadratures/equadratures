from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from math import exp

class TestQR(TestCase):

    def testbasic(self):
        # blackbox
        def blackbox(x):
            return x

        #-----------------------------------------------#
        # analytical distributions 
        # 1) Arcsine distribution --> Chebyshev

        # support ---> a < b , -\infty. +\infy
        a = 0.0#0.001
        b = 1.0#0.99
        x = np.linspace(a, b, 100) # domain for Chebyshev
        mean_1 = (a+b)/2.0 
        variance_1 = (1.0/8.0)*(b-a)**2

        f_X= np.zeros(len(x))

        for i in range(0,len(x)):
            if x[i] == a :
                f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
            elif x[i] == b:
                f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
            else: 
                f_X[i] = 1.0/(np.pi* np.sqrt((x[i] - a)*(b - x[i])) )

        #print f_X
        print 'analytical mean of arcsine:', mean_1
        print 'analytical variance of arcsine:', variance_1

        #----------- effective quadrature -----------------#

        xo = Parameter(order=5, distribution='Chebyshev',lower =0.001, upper=0.99)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean
        print 'Effective quadrature variance:' , myStats.variance
        #------------------------------------------------------------------------------------------------#
        a1,b1 = xo.getPDF(150)
        fig = plt.figure(1)
        #ax = fig.add_subplot(1,1,1)
        #plt.hist(, normed= True, bins=30, range=(0.0,1.0))
        #count, bins, ignored = plt.hist(c1, 30, normed = True)
        plt.plot(x, f_X, 'r-', label='arcsine analyt.')
        plt.plot(a1,b1, 'bo', label='effect.quad')
        plt.legend(loc='upper center')
        #adjust_spines(ax, ['left', 'bottom']) 
        ax = plt.gca()
        ax.set_xlabel('X variable')
        ax.set_ylabel('PDF')
        ax.set_ylim([0.0,3.2])
        plt.show()

        print '--------------------------------------------------------------------------'
        #-----------------------------------------------#
        #2)   Truncated Gaussian

        # support ---> a < b
        # a,b real numbers, sigma^2 >= 0
        print 'truncated gaussian has to be completed'
        print '#---------------------------------------------------------------'
        ## chi l'ha pensata amava complicarsi la vita 0.o

        #parameter:
        mu = 1.0
        sigma = 1.0
        a = 2.0
        b = 4.0
        alpha = (a-mu)/ sigma
        beta = (b-mu)/sigma
        ## PHI(beta) = integral_(0)^(3/sqrt(2)) ( e^(-x^2) )=0.88384 # see wolframalpha
        ## PHI(alpha) = integral_(0)^(1/sqrt(2)) ( e^(-x^2) ) = 0.605018 # see wolframalpha
        Z = 0.5*(1.0+ 2.0/(np.sqrt(np.pi)))*(0.88384) - 0.5*(1.0+ 2.0/(np.sqrt(np.pi)))*0.605018

        x = np.linspace(0,1,150)
        fX = np.linspace(0,1,150)
        #
        for i in range(0,len(x)):
            xi = (x[i]-mu)/sigma
            phi_zeta = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*xi**2)
            fX[i] = phi_zeta/(sigma*Z) 
        #
        ##plt.figure(2)
        ##plt.grid()
        ##plt.plot(x,f_X, 'r-',label='analyt.trunc.normal')
        ##plt.show()
        #
        phi_alpha = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*alpha**2)
        phi_beta = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*beta**2)
        #
        mean = mu + (phi_alpha - phi_beta)*sigma/Z
        variance = (sigma**2)* (1+  (alpha*phi_alpha - beta*phi_beta)/Z  -((alpha*phi_alpha - beta*phi_beta)/Z )**2)
        #
        print 'analytical mean of truncated:', mean
        print 'analytical variance of truncated:', variance
        #
        # #----------- effective quadrature -----------------#
        #
        shape_A = 1.0
        shape_B = 1.0

        xo = Parameter(order=5, distribution='truncated-gaussian',lower =2.0, upper=4.0, shape_parameter_A = shape_A, shape_parameter_B = shape_B )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
        print 'Effective quadrature variance:' , myStats.variance

        #at1,bt1 = xo.getPDF(150)
        #fig = plt.figure(2)
        ##ax = fig.add_subplot(1,1,1)
        ##plt.hist(, normed= True, bins=30, range=(0.0,1.0))
        ##count, bins, ignored = plt.hist(c1, 30, normed = True)
        #plt.plot(x, f_X, 'r-', label='trunc analyt.')
        #plt.plot(a1,b1, 'bo', label='effect.quad')
        #plt.legend(loc='upper center')
        ##adjust_spines(ax, ['left', 'bottom']) 
        #ax = plt.gca()
        #ax.set_xlabel('X variable')
        #ax.set_ylabel('PDF')
        ##ax.set_ylim([0.0,3.2])
        #plt.show()


        print '--------------------------------------------------------------------------'

        #-----------------------------------------------#
        #3)   Beta

        # support : x [0,1] | x (0,1)
        # shape_parameters a,b > 0
        shape_A = 2.0 # alpha
        shape_B = 3.0 # beta
        x = np.linspace(0,1,100)
        mean_3  = shape_A/(shape_A + shape_B)
        variance_3 = (shape_A* shape_B)/((shape_A+shape_B+1)*(shape_A+shape_B)**2)
        c_1= 1.0/12
        c_2 = 2.5058

        for i in range(0,len(x)):
            f_X[i] = (1.0/c_1)* ((x[i])**(shape_A-1))*(1-x[i])**(shape_B-1)
    
        print 'analytical mean of beta distribution:', mean_3
        print 'analytical variance of beta:', variance_3
        #plt.figure(3)
        #plt.grid()
        #plt.plot(x, f_X, '-r', label='beta')
        #plt.legend(loc='upper right')
        #plt.show()

        #----------- effective quadrature -----------------#

        xo = Parameter(order=5, distribution='Beta',lower =0.0, upper=1.0, shape_parameter_A = shape_A, shape_parameter_B = shape_B )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
        print 'Effective quadrature variance:' , myStats.variance


        a,b = xo.getPDF(80)
        fig = plt.figure(3)
        ax = fig.add_subplot(1,1,1)
        #count, bins, ignored = plt.hist(x, 30, normed = True)
        plt.plot(x, f_X, '-r', label='analyt.beta dist.')
        plt.plot(a,b, 'bo', label='eff.quad. beta distr.')
        #adjust_spines(ax, ['left', 'bottom']) 
        plt.legend(loc='upper right')
        ax.set_xlabel('X variable')
        ax.set_ylabel('PDF')
        plt.show()

        print '--------------------------------------------------------------------------'

        #-----------------------------------------------#
        #4)   Cauchy distribution

        # support -\infty + \infty
        # parameters : x0 (location), gamma (scale)>0, both REAL
        x0 = 0.0
        gamma = 0.5
        x = np.linspace(-5,5,100)

        for i in range(0,len(x)):
            f_X[i] = 1.0/(np.pi*gamma*(1+(((x[i]-x0)/gamma)**2)))

        #plt.figure(4)
        #plt.grid()
        #plt.plot(x, f_X, '-r', label='cauchy')
        ##plt.legend(loc='center ')
        #plt.show()

        print 'Cauchy:'
        print 'analytical: mean and variance are not defined in the Real domain'
        print 'in the small intervall -5,5:'
        print 'analytical solution:'
        # insert in wolframalpha the following line:
        # integral_(-5)^(5) ( x/(pi*(1/2)*(1+((x-0)/(0.5))^2 )
        mean = 0
        # insert in wolframalpha the following line:
        # integral_(-5)^(5) ( (x-0)^2 /(pi*(1/2)*(1+((x-0)/(0.5))^2 )
        variance = 1.35741 
        print 'mean of cauchy:', mean
        print 'variance of cauchy', variance

        xo = Parameter(order=5, distribution='Cauchy',lower =-5.0, upper=5.0, shape_parameter_A = x0, shape_parameter_B = gamma )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
        print 'Effective quadrature variance:' , myStats.variance

        a,b = xo.getPDF(150)
        fig = plt.figure(4)
        ax = fig.add_subplot(1,1,1)
        #count, bins, ignored = plt.hist(x, 30, normed = True)
        plt.plot(x, f_X, '-r', label='analyt.Cauchy dist.')
        plt.plot(a,b, 'bo', label='eff.quad. Cauchy distr.')
        #adjust_spines(ax, ['left', 'bottom']) 
        #plt.legend(loc='upper right')
        ax.set_xlabel('X variable')
        ax.set_ylabel('PDF')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()

        print '--------------------------------------------------------------------------'

        #-----------------------------------------------#
        #5)   Gamma distribution

        # suport (0, \infty)
        x = np.linspace(0, 20, 100)
        # parameters: alpha >0, beta >0
        k = 2.0
        theta = 0.9
        # the following costant is the analytical solution of integral_(0)^(\infty) ( x^1 * e^(-x) )
        c_1 = 1.0

        mean_5 = k*theta
        variance_5 = k*theta**2
        print 'mean of gamma:', mean_5
        print 'variance of gamma:', variance_5

        for i in range(0,len(x)):
            f_X[i] = (1/c_1)*(1/theta**k)*((x[i])**(k-1))*np.exp(-x[i]/theta)

        #plt.figure(5)
        #plt.grid()
        #plt.plot(x, f_X, 'k-', label='gamma')
        #plt.legend(loc='center ')
        #plt.show()

        xo = Parameter(order=5, distribution='Gamma',lower =0.0, upper=20.0, shape_parameter_A = 2.0, shape_parameter_B = 0.9 )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
        print 'Effective quadrature variance:' , myStats.variance

        a,b = xo.getPDF(150)
        fig = plt.figure(5)
        ax = fig.add_subplot(1,1,1)
        #count, bins, ignored = plt.hist(x, 30, normed = True)
        plt.plot(x, f_X, '-r', label='analyt.gamma dist.')
        plt.plot(a,b, 'bo', label='eff.quad. gamma distr.')
        #adjust_spines(ax, ['left', 'bottom']) 
        plt.legend(loc='upper right')
        ax.set_xlabel('X variable')
        ax.set_ylabel('PDF')
        plt.show()

        print '--------------------------------------------------------------------------'

            #-----------------------------------------------#
        #6)   Weibul distribution

        #support : [0, \infty)
        x = np.linspace(0.0001,50,100)
        # parameters : scale lambda (0, \infty); shape k (0, \infty)

        lambdaa = 9.0
        k = 0.5

        for i in range(0,len(x)):
            f_X[i] = (k/lambdaa)*((x[i]/lambdaa)**(k-1))* np.exp(-(x[i]/lambdaa)**k)

        # calculated on wolframalpha: 9* integral_(0)^(\infty) ( x^(1/0.5)* e^(-x) ) = 9*2
        mean = 18  # wolframalphai
        # 81*( (integral_(0)^(\infty) ( x^(2/0.5)* e^(-x) ) - [integral_(0)^(\infty) ( x^(1/0.5)* e^(-x) ) ]^2 ) = 81 * (24-4)
        variance = 81.0*(20.0)
        print 'analtyical mean of Weibul:' , mean
        print 'analytical variance of weibul', variance

        #plt.figure(6)
        #plt.grid()
        #plt.plot(x, f_X, 'g-', label='weibul')
        #plt.legend(loc='center ')
        #plt.show()

        xo = Parameter(order=5, distribution='Weibull',lower =0.0001, upper=50.0, shape_parameter_A =lambdaa , shape_parameter_B = k )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'Effective quadrature mean: ', myStats.mean #, myStats.variance
        print 'Effective quadrature variance:' , myStats.variance

        a,b = xo.getPDF(1000)
        fig = plt.figure(6)
        ax = fig.add_subplot(1,1,1)
        #count, bins, ignored = plt.hist(x, 30, normed = True)
        plt.plot(x, f_X, '-r', label='analyt.weibull dist.')
        plt.plot(a,b, 'bo', label='eff.quad. weibull distr.')
        #adjust_spines(ax, ['left', 'bottom']) 
        plt.legend(loc='upper right')
        ax.set_xlabel('X variable')
        ax.set_ylabel('PDF')
        ax.set_ylim([0.0,1.2])
        ax.set_xlim([-1.0,30])
        plt.show()
        print 'done!'

if __name__ == '__main__':
    unittest.main()
