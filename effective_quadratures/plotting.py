#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import matplotlib as mpl


def bestfit(x_train, y_train, x_test, y_test, x_label, y_label, filename=None):
    opacity = 0.8
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.scatter(x_train, y_train, marker='o', s=120, alpha=opacity, color='orangered',linewidth=1.5)
    plt.plot(x_test, y_test, linestyle='-', linewidth=2, color='steelblue')
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def lineplot(x, y, x_label, y_label, filename=None):
    opacity = 0.8
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.plot(x, y, linestyle='-', linewidth=3, color='deepskyblue')
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def scatterplot(x, y, z, filename=None):
    m, n = x.shape
    p, q = y.shape
    if n > m :
        raise(ValueError, 'scatterplot(x, y): Matrix x of size m-by-n, must satisfy m>=n')
    if m is not p:
        raise(ValueError, 'scatterplot(x, y): The number of rows in x must be equivalent to the number of rows in y')
    
        opacity = 0.8
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        mpl.rcParams['axes.linewidth'] = 2.0
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.grid()
        ax.set_axis_bgcolor('whitesmoke')
        plt.scatter(x, y, marker='s', s=70, alpha=opacity, color='limegreen',linewidth=1.5)
        ax.set_axisbelow(True)
        adjust_spines(ax, ['left', 'bottom'])
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
        plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
        else:
            plt.show()

def scatterplot(x, y, filename=None):
    m, n = x.shape
    p, q = y.shape
    if n > m :
        raise(ValueError, 'scatterplot(x, y): Matrix x of size m-by-n, must satisfy m>=n')
    if m is not p:
        raise(ValueError, 'scatterplot(x, y): The number of rows in x must be equivalent to the number of rows in y')
 
    opacity = 0.8
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.scatter(x, y, marker='s', s=70, alpha=opacity, color='limegreen',linewidth=1.5)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def histogram(samples, x_label, y_label, filename=None):
    opacity = 1.0
    error_config = {'ecolor': '0.3'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.hist(samples, 30, normed=1, facecolor='crimson', alpha=opacity)
    plt.xlim(0.08*np.min(samples), 1.2*np.max(samples))
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def barplot(x, y, x_label, y_label, filename=None):
    bar_width = 0.35
    opacity = 1.0
    error_config = {'ecolor': '0.3'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.bar(x, y, bar_width, alpha=opacity, color='steelblue',error_kw=error_config, linewidth=1.5)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='y', colors='black', width=2)
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='x', colors='black', width=2)
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])