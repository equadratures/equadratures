import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from cycler import cycler


def errorplot2D(errors, x_label=None, y_label=None, xlim=None, ylim=None, filename=None):
    G = np.log10(np.abs(errors))
    Zm = np.ma.masked_where(np.isnan(G),G)
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    #ax.set_axis_bgcolor('whitesmoke')
    ax.set_facecolor('whitesmoke')
    plt.pcolor(errors, cmap= cm.jet, vmin=-14, vmax=1)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    
    if x_label is not None:
        plt.xlabel(x_label, fontsize=16)
    
    if y_label is not None:
        plt.ylabel(y_label, fontsize=16)
    
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    cbar = plt.colorbar(extend='neither', spacing='proportional',
                orientation='vertical', shrink=0.8, format="%.0f")
    cbar.ax.tick_params(labelsize=16) 
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')



def coeffplot2D(coefficients, index_set, x_label, y_label, filename=None, vmin_log=None, vmax_log=None):
    
    if vmin_log is None:
        vmin_log = -16
    if vmax_log is None:
        vmax_log = 0

    #elements_used = index_set.elements
    x, y, z, max_order = twoDgrid(coefficients, index_set)
    G = np.log10(np.abs(z))
    Zm = np.ma.masked_where(np.isnan(G),G)
    opacity = 0.8
    plt.rc('font', family='serif')
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.pcolor(y,x, Zm, cmap= cm.jet, vmin=vmin_log, vmax=vmax_log)
    plt.xlim(0, max_order)
    plt.ylim(0, max_order)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(extend='neither', spacing='proportional',
                orientation='vertical', shrink=0.8, format="%.0f")
    cbar.ax.tick_params(labelsize=16) 
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')


def bestfit(x_train, y_train, x_test, y_test, CI, x_label, y_label, filename=None):

    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')

    #plt.fill_between(xo, bottom, top, facecolor='crimson', alpha=0.5)
    patches = []
    for i in range(0, len(y_test)-1):
        xy = np.array([ [x_test[i,0], y_test[i,0] - CI[i]], [x_test[i,0], y_test[i,0] + CI[i]], [x_test[i+1,0], y_test[i+1,0] + CI[i,0]], [x_test[i+1,0], y_test[i+1,0] - CI[i,0]]] )
        #xy = np.random.rand(4,2)
        polygon = Polygon(xy, closed=False)
        patches.append(polygon)
    p = PatchCollection(patches, alpha=0.2, edgecolor='none', facecolor='teal')
    ax.add_collection(p)
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

def bestfit3D(x_train, y_train, x_test, y_test, x_label, y_label, z_label, filename=None):
    m, n = x_train.shape
    p, q = y_train.shape
    if n > m :
        raise(ValueError, 'bestfit3D(x, y): Matrix x of size m-by-n, must satisfy m>=n')
    if m is not p:
        raise(ValueError, 'bestfit3D(x, y): The number of rows in x must be equivalent to the number of rows in y')
    xx1 = x_test[0]
    xx2 = x_test[1]
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rc('axes', edgecolor='white', labelcolor='black', grid=True)
    mpl.rc('xtick', color='black')
    mpl.rc('ytick', color='black')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.w_yaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.w_zaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.w_xaxis.line.set_linewidth(2)
    ax.w_yaxis.line.set_linewidth(2)
    ax.w_zaxis.line.set_linewidth(2)
    plt.grid()
    ax.w_xaxis.gridlines.set_lw(2.0)
    ax.w_yaxis.gridlines.set_lw(2.0)
    ax.w_zaxis.gridlines.set_lw(2.0)
    ax.w_xaxis._axinfo.update({'grid' : {'color': (1.0, 1.0, 1.0, 1)}})
    ax.w_yaxis._axinfo.update({'grid' : {'color': (1.0, 1.0, 1.0, 1)}})
    ax.w_zaxis._axinfo.update({'grid' : {'color': (1.0, 1.0, 1.0, 1)}})
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    for i in range(0, m):
        ax.scatter(x_train[i,0], x_train[i,1], y_train[i],  marker='H', s=90, alpha=opacity, color='teal',linewidth=1.5)
    ax.plot_surface(xx1,xx2, y_test,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.02, alpha=0.6)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def parameterplot(x_axis, y_pdf, y_cdf, filename=None, x_label=None, y_label1=None, y_label2=None):
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0

    fig = plt.figure()
    fig.subplots_adjust(hspace=.25)
    ax = fig.add_subplot(2, 1, 1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    plt.plot(x_axis, y_pdf, linestyle='-', linewidth=3, color='deepskyblue')
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    #plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label1, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    #plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xticklabels([])
    # subplot 1
    ax2 = plt.subplot(2, 1, 2)
    plt.grid()
    ax2.set_axis_bgcolor('whitesmoke')
    plt.plot(x_axis, y_cdf, linestyle='-', linewidth=3, color='crimson')
    ax2.set_axisbelow(True)
    adjust_spines(ax2, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label2, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def semilogy_lineplot(x, y, x_label, y_label, filename=None):
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_facecolor('whitesmoke')
    plt.semilogy(x, y, linestyle='-', linewidth=3, color='deepskyblue')
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
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_facecolor('whitesmoke')
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

def contour_plot(x, y , z, filename=None, pts=None, other_pts=None, path_points = None):

    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_facecolor('whitesmoke')
    K = plt.contourf(x, y, z, 100, rasterized=False) 

    if pts is not None:
        m, n = pts.shape  
        for i in range(0, m):
            plt.scatter(pts[i,0], pts[i,1], marker='o', s=80, color='crimson', linewidth=1.5, edgecolor='black')
    
    if other_pts is not None:
        r, s = other_pts.shape
        for j in range(0, r):
            plt.scatter(other_pts[j,0], other_pts[j,1],  marker='x', s=50, alpha=opacity, color='limegreen', linewidth=3.5)

    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(r'$\zeta_1$', fontsize=16)
    plt.ylabel(r'$\zeta_2$', fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(K)
    cbar.ax.tick_params(labelsize=16) 
    plt.tight_layout()
    
    # Superimpose a path object (if exists)
    if not(path_points is None):
        vertices = path_points.copy()
        codes = [Path.MOVETO]
        for i in range(1, vertices.shape[0]):
            codes.append(Path.LINETO)
        assert len(codes) == vertices.shape[0]
        path = Path(vertices, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor = 'red', lw=2)
        ax.add_patch(patch)
        
    
    if not filename is None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def scatterplot3D(x, f, x1_label=None, x2_label=None, f_label=None, filename=None):
    m, n = x.shape
    p, q = f.shape
    if n > m :
        raise(ValueError, 'scatterplot(x, y): Matrix x of size m-by-n, must satisfy m>=n')
    if m != p:
        raise(ValueError, 'scatterplot(x, y): The number of rows in x must be equivalent to the number of rows in y')
    
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rc('axes', edgecolor='white', labelcolor='black', grid=True)
    mpl.rc('xtick', color='black')
    mpl.rc('ytick', color='black')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, m):
        ax.scatter(x[i,0], x[i,1], f[i],  marker='H', s=90, alpha=opacity, color='darkorange',linewidth=1.5)
    ax.w_xaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.w_yaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.w_zaxis.set_pane_color((0.961, 0.961, 0.961, 1.0))
    ax.set_facecolor('whitesmoke')
    
    if not x1_label is None:
        ax.set_xlabel(x1_label)
    if not x2_label is None:
        ax.set_ylabel(x2_label)
    if not f_label is None:
        ax.set_zlabel(f_label)   
    plt.tight_layout()
    if not filename is None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def scatterplot(x, y, x_label, y_label, filename=None, marker_type=None, color_choice=None):
    x = np.mat(x)
    y = np.mat(y)
    m, n = x.shape
    p, q = y.shape
    if n > m:
        x = x.T
        y = y.T
        m = n

    if marker_type is None:
        marker_type = 's'
    if color_choice is None:
        color_choice = 'limegreen'
   
    opacity = 0.8
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_facecolor('whitesmoke')
    for i in range(0, m):
        plt.scatter(x[i,0], y[i,0], marker=marker_type, s=140, alpha=opacity, color=color_choice,linewidth=1.5)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlim(np.min(x)-0.5, np.max(x)+0.5)
    #plt.ylim(np.min(y)-0.5, np.max(y)+0.5)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
        
def scatterplot2(x, y, x_label, filename=None, marker_type=None):
    # x is m by n where m is the number of points for each series and n is number of series. (each col constitutes a series)
    # y is also m by n
    x = np.mat(x)
    y = np.mat(y)
    assert x.shape == y.shape
    m,n = x.shape

    if marker_type is None:
        marker_type = 's'
 
    opacity = 1.0
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    cm = plt.get_cmap('tab20')
    ax.set_prop_cycle(cycler('color', [cm(i) for i in np.linspace(0, 1,x.shape[1])]))
    ax.set_axis_bgcolor('whitesmoke')
    for i in range(n):    
        plt.scatter([x[:,i]], [y[:,i]], marker=marker_type, s=20, alpha=opacity)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
#    plt.xlabel(x_label, fontsize=3)
#    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
    plt.xticks(x[:,0], x_label, fontsize=8, rotation = 30)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')

def histogram(samples, x_label, y_label, filename=None):
    opacity = 1.0
    error_config = {'ecolor': '0.3'}
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_facecolor('whitesmoke')
    plt.hist(samples, 30, normed=1, facecolor='saddlebrown', alpha=opacity)
    std_dev = np.std(samples)
    plt.xlim(np.min(samples) - 1.2* std_dev, np.max(samples) + 1.2*std_dev)
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

def barplot(x, y, x_label, y_label, x_ticks, filename=None):
    bar_width = 0.35
    opacity = 1.0
    error_config = {'ecolor': '0.3'}
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
    plt.xticks(x, x_ticks, fontsize=16)
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


def twoDgrid(coefficients, index_set):

    # First determine the maximum tensor grid order!
    max_order = int( np.max(index_set) ) + 1

    # Now create a tensor grid with this max. order
    x, y = np.mgrid[0:max_order, 0:max_order]
    z = (x*0 + y*0) + float('NaN')
    counter = 0

    for counter in range(0, len(coefficients)):
        for i in range(0, max_order):
            for j in range(0, max_order):
                if (i == index_set[counter, 0]) and (j == index_set[counter, 1]) : 
                    z[i,j] = coefficients[counter]
                    break

                    
    return x,y,z, max_order

#plot 3 series together
#assume that the x's are shared.
def triplebarplot(x, y1, y2, y3, x_label, y_label, x_ticks, filename=None):
    bar_width = 0.2
    opacity = 1.0
    error_config = {'ecolor': '0.3'}
    mpl.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.grid()
    ax.set_axis_bgcolor('whitesmoke')
    bar1 = plt.bar(x, y1, bar_width, alpha=opacity, color='steelblue',error_kw=error_config, linewidth=1.5)
    bar2 = plt.bar(x + bar_width*np.ones((len(x))), y2, bar_width, alpha=opacity, color='red',error_kw=error_config, linewidth=1.5)
    bar3 = plt.bar(x+ 2*bar_width*np.ones((len(x))), y3, bar_width, alpha=opacity, color='yellow',error_kw=error_config, linewidth=1.5)
    ax.set_axisbelow(True)
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(b=True, which='major', color='w', linestyle='-', linewidth=2)
    plt.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks(x + np.ones((len(x))) * 1.5 * bar_width, x_ticks, fontsize=12)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.legend((bar1[0], bar2[0], bar3[0]), ("Variance", "Skewness", "Kurtosis"))
    if filename is not None:
        plt.savefig(filename, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
def piechart(labels, sizes, title):
    fig1, ax1 = plt.subplots()
    cm = plt.get_cmap('tab20')
    ax1.set_prop_cycle(cycler('color', [cm(i) for i in np.linspace(0, 1,len(labels))]))
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    patches, texts = ax1.pie(sizes, shadow=True, startangle=90)
    
    ax1.axis([0,1.2,-1,1])
    plt.legend(patches, labels, loc='best')
    plt.title(title)
    plt.show()