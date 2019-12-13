from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
VALUE = 15
plt.rcParams.update({'font.size': VALUE})

order_parameters = 3
mass = Parameter(distribution='uniform', lower=30.0, upper=60.0, order=order_parameters)
area = Parameter(distribution='uniform', lower=0.005, upper=0.020, order=order_parameters)
volume = Parameter(distribution='uniform', lower=0.002, upper=0.010, order=order_parameters)
spring = Parameter(distribution='uniform', lower=1000., upper=5000., order=order_parameters)
pressure = Parameter(distribution='uniform', lower=90000., upper=110000., order=order_parameters)
ambtemp = Parameter(distribution='uniform', lower=290., upper=296., order=order_parameters)
gastemp = Parameter(distribution='uniform', lower=340., upper=360., order=order_parameters)
parameters = [mass, area, volume, spring, pressure, ambtemp, gastemp]

def piston(x):
        mass, area, volume, spring, pressure, ambtemp, gastemp = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        A = pressure * area + 19.62*mass - (spring * volume)/(1.0 * area)
        V = (area/(2*spring)) * ( np.sqrt(A**2 + 4*spring * pressure * volume * ambtemp/gastemp) - A)
        C = 2 * np.pi * np.sqrt(mass/(spring + area**2 * pressure * volume * ambtemp/(gastemp * V**2)))
        return C

mybasis = Basis('total-order')
mypoly  = Poly(parameters, mybasis, method='least-squares',sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'qr', 'sampling-ratio':1.0})

print(mypoly.basis.cardinality)
mypoly.set_model(piston)

mean, var = mypoly.get_mean_and_variance()
sobol = mypoly.get_sobol_indices(1)
for i in range(0, len(parameters)):
        print(float(sobol[(i,)]) * 10**2 * var)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
data_1 = np.arange(7) - 0.10 + 1
for i in range(0, len(sobol)):
    plt.bar(i+1, sobol[(i,)], color='steelblue',linewidth=1.5)
ax.set_axisbelow(True)
#adjust_spines(ax, ['left', 'bottom'])
plt.xlabel(r'Parameters', fontsize=16)
plt.ylabel(r"First order Sobol' indices", fontsize=16)
xTickMarks = [r'$M$', r'$S$', r'$V_0$', r'$k$', r'$P_0$', r'$T_a$', r'$T_0$']
ax.set_xticks(data_1+0.10)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=16)
plt.tight_layout()
plt.savefig('../Figures/tutorial_9_fig_a.png', dpi=300, bbox_inches='tight',pad_inches=0.1)

sobol_2nd = mypoly.get_sobol_indices(2)
for key, value in sobol_2nd.items():
    print(str('Parameter numbers: ')+str(key)+", Sobol' index value: "+str(value))
