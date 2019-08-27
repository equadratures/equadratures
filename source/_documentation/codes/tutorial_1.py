from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
VALUE = 15
plt.rcParams.update({'font.size': VALUE})

s = Parameter(distribution='normal', shape_parameter_A = 0.0, shape_parameter_B = 1.0, order=3)
s_values, pdf = s.get_pdf()
s_values, cdf = s.get_cdf()
s_samples = s.get_samples(6000)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axisbelow(True)
plt.plot(s_values, pdf, '-', c='crimson', lw=4)
plt.xlabel('$s$', fontsize=VALUE)
plt.ylabel('PDF', fontsize=VALUE)
plt.fill_between(s_values,  pdf*0.0, pdf, color="crimson" , interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
plt.grid()
plt.savefig('../Figures/tutorial_1_fig_a.png', dpi=200, bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axisbelow(True)
plt.plot(s_values, cdf, '-', c='navy', lw=4)
plt.xlabel('$s$', fontsize=VALUE)
plt.ylabel('CDF', fontsize=VALUE)
plt.grid()
plt.savefig('../Figures/tutorial_1_fig_b.png', dpi=200, bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
n, bins, patches = plt.hist(s_samples, 50, normed=1, facecolor='navy', alpha=0.75, edgecolor='w')
plt.xlabel('$s$', fontsize=1)
plt.ylabel('PDF (Histogram)', fontsize=VALUE)
plt.grid()
ax.set_axisbelow(True)
plt.savefig('../Figures/tutorial_1_fig_c.png', dpi=200, bbox_inches='tight')

s = Parameter(distribution='truncated-gaussian', lower=-1.0, upper=2., shape_parameter_A = 0.0, shape_parameter_B = 1.0, order=3)
s_values, pdf = s.get_pdf()
s_values, cdf = s.get_cdf()
s_samples = s.get_samples(6000)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axisbelow(True)
plt.plot(s_values, pdf, '-', c='crimson', lw=4)
plt.fill_between(s_values,  pdf*0.0, pdf, color="crimson" , interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
plt.xlabel('$s$', fontsize=VALUE)
plt.ylabel('PDF', fontsize=VALUE)
plt.grid()
plt.savefig('../Figures/tutorial_1_fig_d.png', dpi=200, bbox_inches='tight')


# Create some random data sets and add them together!
param1 = np.random.rand(1000)
param2 = np.random.randn(1200)
param3 = np.random.randn(1300)*0.5 - 0.2
param4 = np.random.randn(300)*0.1 + 3
data = np.hstack([param1, param2, param3, param4])
s = Parameter(distribution='custom', data=data, order=3)
s_values, pdf = s.get_pdf()
s_values, cdf = s.get_cdf()
s_samples = s.get_samples(6000)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axisbelow(True)
plt.plot(s_values, pdf, '-', c='crimson', lw=4)
plt.fill_between(s_values,  pdf*0.0, pdf, color="crimson" , interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
plt.hist(data, 100, normed=1, facecolor=None, alpha=0.7, edgecolor='k')
plt.xlabel('$s$', fontsize=VALUE)
plt.ylabel('PDF', fontsize=VALUE)
plt.grid()
plt.savefig('../Figures/tutorial_1_fig_e.png', dpi=200, bbox_inches='tight')
