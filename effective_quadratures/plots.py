#!/usr/bin/env python
"""Utilities for plotting"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



def scatter_plot(X, Y, Xsurf=None, Ysurf=None):
    if dimensions == 1:
        if Xsurf is None:
            
        else:
            
    if dimensions == 2:
        if Xsurf is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:,0].A1, X[:,1].A1, Y.A1, c='red', marker='o', s=120, alpha=0.3)
            ax.view_init(elev=31., azim=168)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=31., azim=168)
            ax.plot_surface(xx1,xx2, yy1,rstride=1, cstride=1, cmap=cm.winter, linewidth=0.02, alpha=0.5)
            ax.scatter(x_train[:,0].A1, x_train[:,1].A1, y_train.A1, c='red', marker='o', s=120, alpha=0.3)
            plt.show()

def pcolor_plot():
    Zm = np.ma.masked_where(np.isnan(z),z)
    plt.pcolor(y,x, Zm, cmap='jet', vmin=-12, vmax=5)
    plt.title('SPAM coefficients')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
