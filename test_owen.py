# Sample test utility!
from equadratures import *
#from parameter import Parameter
#from basis import Basis
#from polyreg import Polyreg
#import utils as utils
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

plt.close("all")
# Function!

p_order = 3

mass = Parameter(order = p_order, lower=30, upper=60)
area = Parameter(order = p_order, lower=0.005, upper=0.020)
volume = Parameter(order = p_order, lower=0.002, upper=0.010)
spring = Parameter(order = p_order, lower=1000, upper=5000)
pressure = Parameter(order = p_order, lower=90000, upper=110000)
ambtemp = Parameter(order = p_order, lower=290, upper=296)
gastemp = Parameter(order = p_order, lower=340, upper=360)
parameters = [ambtemp, gastemp, volume, spring, pressure, mass, area]

#def piston(x):
#    mass, area, volume, spring, pressure, ambtemp, gastemp = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
#    A = pressure * area + 19.62*mass - (spring * volume)/(1.0 * area)
#    V = (area/(2*spring)) * ( np.sqrt(A**2 + 4*spring * pressure * volume * ambtemp/gastemp) - A)
#    C = 2 * np.pi * np.sqrt(mass/(spring + area**2 * pressure * volume * ambtemp/(gastemp * V**2)))
#    return C
def piston(x):
    ambtemp, gastemp, volume, spring, pressure, mass, area = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    A = pressure * area + 19.62*mass - (spring * volume)/(1.0 * area)
    V = (area/(2*spring)) * ( np.sqrt(A**2 + 4*spring * pressure * volume * ambtemp/gastemp) - A)
    C = 2 * np.pi * np.sqrt(mass/(spring + area**2 * pressure * volume * ambtemp/(gastemp * V**2)))
    return C
x0 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x1 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
#x2 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = 7)
#x3 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = 7)
#parameters = [x0,x1,x2,x3]
parameters = [x0,x1]
def G_fun(x):
    return 10*(x[0]-.5)**3 + .5*x[1]**2 + 1.0
#    return ((2*x[0]+1)/2) * ((2*x[1]+1)/2) * ((2*x[2]+1)/2)

#def G_fun(x):
#    f = 1.0
#    for i in range(4):
#        t = (np.abs(4*x[i] - 2) + (i+1)**2.0)/(1 + (i+1)**2.0)
#        f = f * t
#    return f

#orders = [p_order,p_order,p_order,p_order]
#orders = [p_order,p_order,p_order]
orders = [p_order,p_order]
polybasis = IndexSet("Tensor grid", orders)


# get some points to evaluate on
x = np.random.uniform(size = (1500,2))
#x = np.random.uniform(size = (1500,3))
#x = np.random.uniform(size = (1500,7))

#x = x * [30, .015, .008, 4000, 20000, 6, 20]
#x = x + [30, .005, .002, 1000, 90000, 290, 340]

#x = x * [6, 20, .008, 4000, 20000, 30,.015]
#x = x + [290, 340, .002, 1000, 90000, 30, .005]

poly = Polyreg(x, parameters, polybasis, fun=G_fun)

#coeffs = poly.coefficients

s = poly.getStatistics()

print s.mean
print s.variance
print s.skewness
print s.kurtosis

fosi = s.getSobol(1)
#sosi = s.getSobol(2)
#tosi = s.getSobol(3)
#print np.array(fosi.values()) * s.variance * 100

print s.getCondSkewness(1)
print s.getCondSkewness(2)

#print (np.array(s.getSobol(1).values()) * s.variance ) * 100
#print sum(s.getSobol(1).values())
##print s.getCondSkewness(1)
#skew1 = s.getCondSkewness(1)
##skew2 = s.getCondSkewness(2)
##skew3 = s.getCondSkewness(3)
##skew4 = s.getCondSkewness(4)
#TSI_skew = s.calc_TSI([skew1])
k1 = s.getCondKurtosis(1)
k2 = s.getCondKurtosis(2)
k = k1.copy()
k.update(k2)
print sum(k.values())
#TSI_kurt = s.calc_TSI([kurt1,kurt2])
#
#
#print (np.array(TSI_skew) * s.skewness * s.variance**1.5 ) * 10**5
##print sum(skew1.values())+sum(skew2.values())+sum(skew3.values())
#print (np.array(TSI_kurt) * s.kurtosis * s.variance**2 ) * 10**5
#print sum(kurt1.values())+sum(kurt2.values())

#large_number = 100000
#x = np.random.uniform(size=(large_number, 4))
##x = 2.0* x - 1
#
#f = np.zeros((large_number,1))
#for i in range(0, large_number):
#    f[i,0] = G_fun(x[i,:])
#x0_samples = mass.getSamples(100000)
#x1_samples = area.getSamples(100000)
#x2_samples = volume.getSamples(100000)
#x3_samples = spring.getSamples(100000)
#x4_samples = pressure.getSamples(100000)
#x5_samples = ambtemp.getSamples(100000)
#x6_samples = gastemp.getSamples(100000)
#f = np.zeros((100000,1))
#
#for i in range(100000):
#    f[i,0] = piston([x0_samples[i,0], x1_samples[i,0], x2_samples[i,0], x3_samples[i,0], x4_samples[i,0], x5_samples[i,0], x6_samples[i,0]])
#
#print 'MONTE CARLO'
#print np.var(f)
#print st.skew(f)
#print st.kurtosis(f, fisher = False)

#v1 = s.getSobol(1)
#v2 = s.getSobol(2)
#v3 = s.getSobol(3)
#s1 = s.getCondSkewness(1)
#s2 = s.getCondSkewness(2)
#k1 = s.getCondKurtosis(1)
#print k
#k2 = s.getCondKurtosis(2)
#print k2
#s3 = s.getCondSkewness(3)
#k3 = s.getCondKurtosis(3)
#print v1
#print s1
#print k1
#print np.array(s.calc_TSI([v1]))*s.variance*100

# Plot piston as a fn of area, all others fixed

#plot_x = np.zeros((100, 7))
#plot_x[:,0] = 60
#plot_x[:,2] = .010
#plot_x[:,3] = 5000
#plot_x[:,4] = 110000
#plot_x[:,5] = 296
#plot_x[:,6] = 360
#
#plot_x[:,1] = np.linspace(.005, .020, 100)
#
#plot_y = np.apply_along_axis(piston, 1, plot_x)
#plt.figure(0)
#plt.plot(plot_x[:,1], plot_y)
#
#plot_x1 = plot_x.copy()
#
#plot_x1[:,0] = 30
#plot_x1[:,2] = .002
#plot_x1[:,3] = 1000
#plot_x1[:,4] = 90000
#plot_x1[:,5] = 290
#plot_x1[:,6] = 340
#
#plot_y1 = np.apply_along_axis(piston, 1, plot_x1)
#
#plt.plot(plot_x1[:,1], plot_y1)
#
#plot_x2 = plot_x.copy()
#
#plot_x2[:,0] = 45
#plot_x2[:,2] = .006
#plot_x2[:,3] = 3000
#plot_x2[:,4] = 100000
#plot_x2[:,5] = 293
#plot_x2[:,6] = 350
#
#plot_y2 = np.apply_along_axis(piston, 1, plot_x2)
#
#plt.plot(plot_x2[:,1], plot_y2)
#print (plot_y[0] - plot_y[1])/(plot_x[0,1] - plot_x[1,1])
#
#plt.legend(['0','1','2'])
#
## As a fn of volume
#
#
#plot_x[:,0] = 60
#plot_x[:,1] = .02
#plot_x[:,3] = 5000
#plot_x[:,4] = 110000
#plot_x[:,5] = 296
#plot_x[:,6] = 360
#
#plot_x[:,2] = np.linspace(.002, .010, 100)
#
#plot_y = np.apply_along_axis(piston, 1, plot_x)
#plt.figure(1)
#plt.plot(plot_x[:,2], plot_y)
#
#plot_x1 = plot_x.copy()
#
#
#plot_x1[:,0] = 30
#plot_x1[:,1] = .005
#plot_x1[:,3] = 1000
#plot_x1[:,4] = 90000
#plot_x1[:,5] = 290
#plot_x1[:,6] = 340
#
#plot_y1 = np.apply_along_axis(piston, 1, plot_x1)
#
#plt.plot(plot_x1[:,2], plot_y1)
#
#
#plot_x2 = plot_x.copy()
#
#
#plot_x2[:,0] = 45
#plot_x2[:,1] = .0125
#plot_x2[:,3] = 3000
#plot_x2[:,4] = 100000
#plot_x2[:,5] = 293
#plot_x2[:,6] = 350
#
#plot_y2 = np.apply_along_axis(piston, 1, plot_x2)
#print (plot_y2[98] - plot_y2[99])/(plot_x2[98,2] - plot_x2[99,2])
#plt.plot(plot_x2[:,2], plot_y2)

#[30, .005, .002, 1000, 90000, 290, 340]
#plt.figure(0)
#plot_x = np.zeros((100, 2))
#plot_x[:,0] = 0.0
#
#plot_x[:,1] = np.linspace(0.0, 1.0, 100)
#
#plot_y = np.apply_along_axis(G_fun, 1, plot_x)
#
#plt.plot(plot_x[:,1], plot_y)
#
##plt.figure(1)
#plot_x[:,1] = 0.0
#
#plot_x[:,0] = np.linspace(0.0, 1.0, 100)
#
#plot_y = np.apply_along_axis(G_fun, 1, plot_x)
#
#plt.plot(plot_x[:,0], plot_y)
#
#plt.legend(['0','1'])















# Tensor grid check!
#N = 30 # number of evaluations!
#number = 4 # Highest order!
#dimensions = 2 # Number of dimensions!
#params = [] # Initialize list for parameters!
#chosenOrders = []
#param = Parameter(param_type="Gaussian", shape_parameter_A = 0.0, shape_parameter_B = 1.0, order=number)
#
## Fill up params!
#for i in range(0, dimensions):
#    params.append(param)
#    chosenOrders.append(number)
#
#print "done"
## Create training data!
#chosenBasis_0 = Basis(basis_type='Total order', orders=chosenOrders)
#chosenBasis_1 = Basis(basis_type='Tensor grid', orders=chosenOrders)
##chosenBasis = Basis(basis_type='Hyperbolic basis', orders=chosenOrders, q=0.75)
#
#X = np.random.randn(N, dimensions)
#Y = utils.evalfunction(X, fun)
#
##f1 = open("X_data.txt", "r")
##f2 = open("fX_Data.txt", "r")
##
##X = np.zeros((975,24))
##fX = np.zeros((975,3))
##
##for i in range(975):
##    line = f1.readline()[:-1]
##    X[i,:] = filter(None, line.split(' '))
##    line = f2.readline()[:-1]
##    fX[i,:] = filter(None, line.split(' '))
##
##Y = fX[:,0]
#
## Coefficient check!
#P_0 = Polyreg(basis=chosenBasis_0, parameters=params, training_x=X, training_y=Y)
#P_1 = Polyreg(basis=chosenBasis_1, parameters=params, training_x=X, training_y=Y)
#
#F = P_0.get_F_stat(P_0.coefficients, P_0.A, P_1.coefficients, P_1.A, Y)
#print f.cdf(F, 10, 5)

# Function approx check!
#X2 = np.random.rand(1, dimensions)
#r = P.getPolyFit()
#print r(X2)

# Sobol indices check!
#S = P.getStatistics()
#fosi0 = S.getSobol(order=1)
#sosi0 = S.getSobol(order=2)
#print S.skewness
#print S.kurtosis
#fo_skew0 = S.getCondSkewness(order=1)
#so_skew0 = S.getCondSkewness(order=2)

#print "fosi"
#print fosi0
#
#for key, value in sorted(fosi0.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#    
#
#print "sosi"
#print sosi0
#for key, value in sorted(sosi0.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "fo_skew"
#print fo_skew0
#for key, value in sorted(fo_skew0.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "so_skew"
#print so_skew0
#
#for key, value in sorted(so_skew0.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)




# Check polynomial approximation!
#g = P.getPolynomialApproximant()
#print 'Error!'
#print np.linalg.norm(g - Y, 2)

#del P
#del S
#
#Y = fX[:,1]
#
## Coefficient check!
#P = Polyreg(basis=chosenBasis, parameters=params, training_x=X, training_y=Y)
#
#
## Function approx check!
##X2 = np.random.rand(1, dimensions)
##r = P.getPolyFit()
##print r(X2)
#
## Sobol indices check!
#S = P.getStatistics()
#fosi1 = S.getSobol(order=1)
#sosi1 = S.getSobol(order=2)
#print S.skewness
#print S.kurtosis
#fo_skew1 = S.getCondSkewness(order=1)
#so_skew1 = S.getCondSkewness(order=2)
#
#print "fosi"
#print fosi1
#
#for key, value in sorted(fosi1.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#    
#
#print "sosi"
#print sosi1
#for key, value in sorted(sosi1.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "fo_skew"
#print fo_skew1
#for key, value in sorted(fo_skew1.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "so_skew"
#print so_skew1
#
#for key, value in sorted(so_skew1.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#
#del P
#del S
#Y = fX[:,2]
#P = Polyreg(basis=chosenBasis, parameters=params, training_x=X, training_y=Y)
#
#S = P.getStatistics()
#fosi2 = S.getSobol(order=1)
#sosi2 = S.getSobol(order=2)
#print S.skewness
#print S.kurtosis
#fo_skew2 = S.getCondSkewness(order=1)
#so_skew2 = S.getCondSkewness(order=2)
#
#print "fosi"
#print fosi2
#
#for key, value in sorted(fosi2.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#    
#
#print "sosi"
#print sosi2
#for key, value in sorted(sosi2.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "fo_skew"
#print fo_skew2
#for key, value in sorted(fo_skew2.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#print "so_skew"
#print so_skew2
#
#for key, value in sorted(so_skew2.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)



