import numpy as np
import matplotlib.pyplot as plt
from math import exp
from equadratures import *

#--------------------------------------------------#
#       plot of distributions
#--------------------------------------------------#
y = np.linspace(0.0,1,100)
x = np.linspace(0.0,1,100)
#-----------------------------------------------------------#

def blackbox(x):
    # return 1/x
    return x[0]*x[1]
#-----------------------------------------------------------#
#       PLOT DISTRIBUTIONS
f_xy = np.zeros((100))
domain = np.zeros((100))
coord = np.ones((100))
f_z  = np.zeros((100))
z    = blackbox([x,y])
c = 0.05# random value for plot domain

for i in range(0,99):
    #f_y[i] = f_x[i]
    domain[i] = c/(x[i])
    f_xy[i] = x[i] 
    f_z[i] = -np.log(z[i+1])

plt.figure()
plt.grid()
plt.plot(coord,f_xy,'r-')
plt.plot(f_xy,coord,'r-')
plt.plot(x,domain,'b-')
plt.fill_between(x, 0, domain)
plt.legend(loc='upper right')
axes = plt.gca()
axes.set_xlim([0,1.0])
axes.set_ylim([0,1.0])
plt.xlabel('X')
plt.ylabel('Y')
plt.text(0.1,0.1, '$XY<Z$', fontsize=15)
plt.show()

#---------------------------------------------------------------#
#       analytical solution
# see wolframalpha
global EX
global varX
EX =  (1.0/4.0)
print 'analytical E[x]=', EX
varX = (7.0/144.0)
print 'analytical variance', varX
#----------------------------------------------------------------#
#   tolerance : mean and variance values

def tolerance_m(x):
    error = (EX - x)/EX
    return np.abs(error)

def tolerance_v(x):
    error = 100*(varX - x)/varX
    return np.abs(error)
#----------------------------------------------------------------#
# calculation of mean and variance
# MonteCarlo method

# start is the initial number of samples for MOnte Carlo
start = 50000
tolerance = 0.0001

def MonteCarlo_m(n):

    N = n
    #A = 0.5  # shape factor lambda
    #B = 1/A
    xi = np.random.uniform(0,1,(N,1))
    yi = np.random.uniform(0,1,(N,1))
    zi = evalfunction(np.reshape([xi,yi], (N,2)), blackbox)
    #print 'Monte Carlo:'
    return np.mean(zi)

#print 'ingresso montecarlo:', MonteCarlo_m(9000)

def MonteCarlo_v(n):

    N = n
    #A = 0.5  # shape factor lambda
    #B = 1/A
    xi = np.random.uniform(0,1,(N,1))
    yi = np.random.uniform(0,1,(N,1))
    zi = evalfunction(np.reshape([xi,yi], (N,2)), blackbox)
    return np.var(zi) 
#print 'variance mc:', MonteCarlo_v(9000)  

N = start

print '-------------------------------------------------------'
print '             MEAN'

while True:
    print 'calculating mean of output:'
    print 'Monte Carlo starting number of samples:', N
    verify_m = tolerance_m(MonteCarlo_m(N))
    print 'actual number of samples:', N
    print 'error:', verify_m
    N = int(N*(1.25))
    if verify_m < tolerance:
        print 'error < ' , tolerance
        print 'mean calculated:', MonteCarlo_m(N)
        break

print '-------------------------------------------------------'
print '             VARIANCE'
N = start
while True:
    print 'calculating variance of output:'
    print 'Monte Carlo starting number of samples:', N
    verify_v = tolerance_v(MonteCarlo_v(N))
    print 'actual number of samples:', N
    print 'error:', verify_m
    N = int(N*(1.25))
    if verify_m < tolerance:
        print 'error < ', tolerance
        print 'variance calculated:', MonteCarlo_v(N)
        break
     

#--------------------------------------------------------------#
# Effective quadrature
#--------------------------------------------------------------#
start = 2 # starting value of order
O_numb = start

def EffQ_m(x):

    xo = Parameter(order = x, distribution ='Uniform',lower = 0.0, upper=1) 
    yo = Parameter(order = x, distribution ='Uniform',lower = 0.0, upper=1) 

    myBasis = Basis('Tensor')
    myPoly = Polyint([xo,yo], myBasis)
    myPoly.computeCoefficients(blackbox)
    myStats = myPoly.getStatistics()
    #print 'Effective Quadratures'
    #print 'mean:', myStats.mean, 'variance:', myStats.variance
    return myStats.mean

#a = EffQ_m(3)
#print 'da effective quadrature function:' , a 

def EffQ_v(x):

    xo = Parameter(order = x, distribution ='Uniform',lower = 0.0, upper=1) 
    yo = Parameter(order = x, distribution ='Uniform',lower = 0.0, upper=1) 

    myBasis = Basis('Tensor')
    myPoly = Polyint([xo,yo], myBasis)
    myPoly.computeCoefficients(blackbox)
    myStats = myPoly.getStatistics()
    #print 'Effective Quadratures'
    #print 'mean:', myStats.mean, 'variance:', myStats.variance
    return myStats.variance

print '-------------------------------------------------------'
print '      EFFECTIVE QUADRATURE:    MEAN'

while True:
    print 'calculating mean of output:'
    print 'Order:', O_numb
    verify_m = tolerance_m(EffQ_m(O_numb))
    print 'error:', verify_m
    O_numb = O_numb + 1
    if verify_m < tolerance:
        print 'error < ' , tolerance
        print 'mean calculated:', EffQ_m(O_numb)
        break

O_numb = start

print '-------------------------------------------------------'
print '      EFFECTIVE QUADRATURE:    VARIANCE'

while True:
    print 'calculating variance of output:'
    print 'Order:', O_numb
    verify_v = tolerance_v(EffQ_v(O_numb))
    print 'error:', verify_m
    O_numb = O_numb + 1
    if verify_m < tolerance:
        print 'error < ', tolerance
        print 'variance calculated:', EffQ_v(O_numb)
        break

print 'end'

