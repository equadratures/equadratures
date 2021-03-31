"""Plotting utilities."""
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib
from equadratures.datasets import score
import numpy as np
import scipy as sp
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_style("ticks")

def plot_2D_contour_zonotope(mysubspace, minmax=[-3.5, 3.5], grid_pts=180,  \
                             save=False, xlabel='$u_1$', ylabel='$u_2$', xlim=None, ylim=None, \
                             show=True, return_figure=False):
    """
    Generates a 2D contour plot of the polynomial ridge approximation.
    
    :param Subspace mysubspace:
        An instance of the Subspace class.
    :param list minmax:
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    :param int grid_pts:
        The number of grid points for generating the contour plot.
    :param bool save: 
        Option to save the plot as a .png file.
    :param string xlabel:
        The label used on the horizontal axis.
    :param string ylabel:
        The label used on the vertical axis.        
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    :param bool show: 
        Option to view the plot.
    :param bool return_figure: 
        Option to return the figure and axes instances of the ``matplotlib`` classes.
        
    """
    
    # Utilities for contour plot.
    x1 = np.linspace(minmax[0], minmax[1], grid_pts)
    XX, YY = np.meshgrid(x1, x1)
    xx1 = XX.reshape(grid_pts*grid_pts, )
    yy1 = YY.reshape(grid_pts*grid_pts, )
    H = np.vstack([xx1,yy1]).T
    
    A, b = mysubspace.get_linear_inequalities()
    subspacepoly = mysubspace.get_subspace_polynomial()
    H_evals = subspacepoly.get_polyfit(H)
    
    subspace_poly_evals = np.zeros((grid_pts*grid_pts, ))
    for i in range(0, grid_pts*grid_pts):
        u_coord = np.array([H[i, 0], H[i, 1]]).reshape(2, 1)
        b_out = A @ u_coord
        if np.all(b_out.flatten() <= b):
            subspace_poly_evals[i] = H_evals[i]
        else:
            subspace_poly_evals[i] = np.nan
    
    pts = mysubspace.get_zonotope_vertices()
    hull = ConvexHull(pts)
    u = mysubspace.sample_points @ mysubspace._subspace[:,0:2]
    
    # Contour plot!
    fig = plt.figure(figsize=(12, 7), facecolor=None)
    fig.patch.set_alpha(0.)
    ax = fig.add_subplot(111)
    norm = matplotlib.colors.Normalize(vmin=np.min(mysubspace.sample_outputs) - 0.4 * np.std(mysubspace.sample_outputs),\
                                               vmax=np.max(mysubspace.sample_outputs)+ 0.4 * np.std(mysubspace.sample_outputs))
    ax.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False, norm=norm)  
    c = ax.scatter(u[:,0],u[:,1], c=mysubspace.sample_outputs, marker='o', edgecolor='w', lw=1, alpha=0.8, s=80, norm=norm )  
    plt.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for simplex in hull.simplices:
        plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-', lw=5)
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])
    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('plot_2D_zonotope.png', dpi=140, bbox_inches='tight')  
    if show:
        plt.show()
    if return_figure:
        return fig, ax
    
def plot_samples_from_second_subspace_over_first(mysubspace_1, mysubspace_2, no_of_samples=500,\
                             minmax=[-3.5, 3.5], grid_pts=180,  \
                             save=False, xlabel='$u_1$', ylabel='$u_2$', xlim=None, ylim=None, \
                             show=True, return_figure=False): 

    """
    Generates a zonotope plot where samples from the second subspace are projected
    over the first.
    
    
    :param Subspace mysubspace_1:
        An instance of the Subspace class.
    :param Subspace mysubspace_1:
        A second instance of the Subspace class.
    :param int no_of_samples:
        Number of inactive samples to be generated.
    :param list minmax:
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    :param int grid_pts:
        The number of grid points for generating the contour plot.
    :param bool save: 
        Option to save the plot as a .png file.
    :param string xlabel:
        The label used on the horizontal axis.
    :param string ylabel:
        The label used on the vertical axis.        
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    :param bool show: 
        Option to view the plot.
    :param bool return_figure: 
        Option to return the figure and axes instances of the ``matplotlib`` classes.
    """    
    # 1. Generate a grid on the first zonotope.
    x1 = np.linspace(minmax[0], minmax[1], grid_pts)
    XX, YY = np.meshgrid(x1, x1)
    xx1 = XX.reshape(grid_pts*grid_pts, )
    yy1 = YY.reshape(grid_pts*grid_pts, )
    H = np.vstack([xx1,yy1]).T

    A, b = mysubspace_1.get_linear_inequalities()
    subspacepoly_1 = mysubspace_1.get_subspace_polynomial()
    subspacepoly_2 = mysubspace_2.get_subspace_polynomial()
    H_evals = subspacepoly_1.get_polyfit(H)

    subspace_poly_evals = np.zeros((grid_pts*grid_pts, ))
    for i in range(0, grid_pts*grid_pts):
        u_coord = np.array([H[i, 0], H[i, 1]]).reshape(2, 1)
        b_out = A @ u_coord
        if np.all(b_out.flatten() <= b):
            subspace_poly_evals[i] = H_evals[i]
        else:
            subspace_poly_evals[i] = np.nan

    indices = np.argwhere(~np.isnan(subspace_poly_evals))
    pts_inside = H[indices.flatten(),:]
    F = pts_inside.shape[0]

    # 2. For each point on this grid, generate no_of_sample points along the
    # inactive subspace
    mean_values = np.zeros((F, 1))
    std_values = np.zeros((F, 1))
    for counter in range(0, F):
        X_samples = mysubspace_1.get_samples_constraining_active_coordinates(no_of_samples, \
                                            pts_inside[counter, :].reshape(2,))
        y_values_for_samples = subspacepoly_2.get_polyfit(X_samples)
        mean_values[counter] = np.mean(y_values_for_samples)
        std_values[counter] = np.std(y_values_for_samples)

    pts_zono = mysubspace_1.get_zonotope_vertices()
    hull = ConvexHull(pts_zono)
    
    subspace_poly_evals[indices] =  mean_values
    
    # Contour plot!
    fig = plt.figure(figsize=(12, 7), facecolor=None)
    fig.patch.set_alpha(0.)
    ax = fig.add_subplot(111)
    c = ax.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False)  
    plt.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for simplex in hull.simplices:
        plt.plot(pts_zono[simplex, 0], pts_zono[simplex, 1], 'k-', lw=5)
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])
    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('mean.png', dpi=140, bbox_inches='tight')  
    if show:
        plt.show()
        
    subspace_poly_evals[indices] =  std_values
    
    # Contour plot!
    fig = plt.figure(figsize=(12, 7), facecolor=None)
    fig.patch.set_alpha(0.)
    ax = fig.add_subplot(111)
    c = ax.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False)  
    plt.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for simplex in hull.simplices:
        plt.plot(pts_zono[simplex, 0], pts_zono[simplex, 1], 'k-', lw=5)
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])
    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('std.png', dpi=140, bbox_inches='tight')  
    if show:
        plt.show()

def plot_Sobol_indices(Polynomial, order=1, save=False, show=True, return_figure=False, parameters=None):
    """
    Generates a bar chart for Sobol indices.

    :param Poly Polynomial: 
        An instance of the Poly class.
    :param int order:
        Order of the requires sobol indices.
    :param list parameters: 
        List of the parameters for the given polynomial.
    :param bool save: 
        Option to save the plot as a .png file.
    :param bool show: 
        Option to show the graph.
    :param bool return_figure: 
        Option to get the figure axes,figure.

    """
    if (parameters)==None:
        parameters_x=[r'$S_%d$' %i  for i in range(0,Polynomial.dimensions)]
    else:  
        parameters_x=[i for i in parameters]
    sobol_indices=Polynomial.get_sobol_indices(order)
    fig=plt.figure(figsize=(9,9))
    ax=plt.subplot()
    if order==1:
        x=0
        for i in range(len(parameters_x)):
            plt.bar(x,sobol_indices[(i,)],color='green')
            x=x+1
        plt.xlabel(r'Parameters')
        plt.ylabel(r'Sobol Indices for order {}'.format(order))
        xticks=[]
        xticks.append(" ")
        for i in range(len(parameters_x)):
            xticks.append(parameters_x[i])
            ax.set_xticklabels(xticks,Fontsize=20,rotation=45)
    elif order==2:
        x=0
        for i in range(len(parameters_x)):
            for j in range(i+1,len(parameters_x)):
                plt.bar(x,sobol_indices[(i,j)],color='green')
                x=x+1
        plt.xlabel(r'Parameters')
        plt.ylabel(r'Sobol Indices for order {}'.format(order))
        xticks=[]
        for i in range(0,len(parameters_x)):
            for j in range(i+1,len(parameters_x)):
                string=parameters_x[i] +' ' +parameters_x[j]
                xticks.append(string)
        ax.set_xticks(sp.arange(len(sobol_indices)))
        ax.set_xticklabels(xticks)
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=45,Fontsize=10)
    elif order==3:
        x=0
        for i in range(len(parameters_x)):
            for j in range(i+1,len(parameters_x)):
                for k in range(j+1,len(parameters_x)):
                    plt.bar(x,sobol_indices[(i,j,k)],color='green')
                    x=x+1
        plt.xlabel(r'Parameters')
        plt.ylabel(r'Sobol Indices for order {}'.format(order))
        xticks=[]
        for i in range(len(parameters_x)):
            for j in range(i+1,len(parameters_x)):
                for k in range(j+1,len(parameters_x)):
                    string=parameters_x[i]+' '+parameters_x[j]+' '+parameters_x[k]
                    xticks.append(string)
        ax.set_xticks(sp.arange(len(sobol_indices)))
        ax.set_xticklabels(xticks,Fontsize=10,rotation=45)
    if save:
        plt.savefig('sobol_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig,ax
def plot_pdf(Parameter, ax=None, data=None, save=False, xlim=None, ylim=None, show=True, return_figure=False):
    """
    Plots the probability density function for a Parameter.

    :param Parameter Parameter: 
        An instance of the Parameter class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class.
    :param numpy.array data: 
        Samples from the distribution (or a similar one) that need to be plotted as a histogram.
    :param bool save: 
        Option to save the plot as a .png file.
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    """
    s_values, pdf = Parameter.get_pdf()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    plt.fill_between(s_values,  pdf*0.0, pdf, color="gold" , label='Density', interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
    if data is not None:
        plt.hist(data, 50, density=True, facecolor='dodgerblue', alpha=0.7, label='Data', edgecolor='white')
    plt.xlabel(Parameter.variable.capitalize())
    plt.ylabel('PDF')
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])
    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    plt.legend()
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('pdf_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig, ax
def plot_orthogonal_polynomials(Parameter, ax=None, order_limit=None, number_of_points=200, save=False, xlim=None, ylim=None, show=True, return_figure=False):
    """
    Plots the first few orthogonal polynomials.

    :param Parameter Parameter: 
        An instance of the Parameter class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class.
    :param int order_limit:
        The maximum number of orthogonal polynomials that need to be plotted.
    :param int number_of_points: 
        The number of points used for plotting.
    :param bool save: 
        Option to save the plot as a .png file.
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    :param bool show: 
        Option to view the plot.
    :param bool return_figure: 
        Option to return the figure and axes instances of the ``matplotlib`` classes.

    **Example**::

        import numpy as np
        from equadratures import *

        myparam = eq.Parameter(distribution='uniform', lower = -1.0, upper = 1.0, order=8, endpoints='both')
        myparam.plot_orthogonal_polynomials(xlim=[-1.6, 1.6])
        
    """
    Xi = np.linspace(Parameter.distribution.x_range_for_pdf[0], \
                Parameter.distribution.x_range_for_pdf[-1], number_of_points).reshape(number_of_points, 1)
    P, _, _ = Parameter._get_orthogonal_polynomial(Xi)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel(Parameter.name.capitalize()+' parameter')
        ax.set_ylabel('Orthogonal polynomials')
    if order_limit is None:
        max_order = P.shape[0]
    else:
        max_order = order_limit
    for i in range(0, max_order):
        ax.plot(Xi, P[i,:], '-', lw=2, label='order %d'%(i))
    if xlim is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim is not None:
        ax.set_ylim([ylim[0], ylim[1]])
    ax.legend()
    sns.despine(offset=10, trim=True)
    if save:
        fig.savefig('polyfit_1D_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig, ax
def plot_polyfit_1D(Polynomial, uncertainty=True, output_variances=None, number_of_points=200, save=False, xlim=None, ylim=None, show=True, return_figure=False):
    """
    Plots a 1D only polynomial fit to the data.

    :param Poly Polynomial: 
        An instance of the Polynomial class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class.
    :param bool uncertainty:
        Option to show confidence intervals (1 standard deviation).
    :param numpy.array output_variances:
        User-defined uncertainty associated with each data point; can be either a ``float`` in which case all data points are assumed to have the same variance, or can be an array of length equivalent to the number of data points.
    :param bool save: 
        Option to save the plot as a .png file.
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    :param bool show: 
        Option to view the plot.
    :param bool return_figure: 
        Option to return the figure and axes instances of the ``matplotlib`` classes.
        
    """
    if Polynomial.dimensions != 1:
        raise(ValueError, 'plot_polyfit_1D is only meant for univariate polynomials.')
    Xi = np.linspace(Polynomial.parameters[0].distribution.x_range_for_pdf[0], \
                Polynomial.parameters[0].distribution.x_range_for_pdf[-1], number_of_points).reshape(number_of_points, 1)
    if uncertainty:
        if output_variances is None:
            y, ystd = Polynomial.get_polyfit(Xi,uq=True)
        else:
            Polynomial.output_variances = output_variances
            y, ystd = Polynomial.get_polyfit(Xi,uq=True)
        ystd = ystd.squeeze()
    else:
        y = Polynomial.get_polyfit(Xi)
    y = y.squeeze()
    X = Polynomial.get_points()
    y_truth = Polynomial._model_evaluations
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    plt.plot(Xi, y, '-',label='Polynomial fit', color='navy')
    plt.plot(X.flatten(), y_truth.flatten(), 'o', color='dodgerblue', ms=10, markeredgecolor='k',lw=1, alpha=0.6, label='Data')
    if uncertainty:
        ax.fill_between(Xi.flatten(), y+ystd, y-ystd, alpha=.10, color='deepskyblue',label='Polynomial $\sigma$')
    if xlim is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim is not None:
        ax.set_ylim([ylim[0], ylim[1]])
    plt.legend()
    sns.despine(offset=10, trim=True)
    plt.xlabel(Polynomial.parameters[0].variable.capitalize())
    plt.ylabel('Polynomial fit')
    if save:
        plt.savefig('polyfit_1D_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig, ax
def plot_model_vs_data(Polynomial, sample_data=None, metric='adjusted_r2', save=False, xlim=None, ylim=None, show=True, return_figure=False):
    """
    Plots the polynomial approximation against the true data.

    :param Poly self: 
        An instance of the Poly class.
    :param list sample_data:
        A list formed by ``[X, y]`` where ``X`` represents the spatial data input and ``y`` the output.
    :param bool save: 
        Option to save the plot as a .png file.
    :param list xlim: 
        Lower and upper bounds for the horizontal axis, for example ``xlim=[-3, 5]``.
    :param list ylim: 
        Lower and upper bounds for the vertical axis, for example ``ylim=[-1, 1]``.
    :param bool show: 
        Option to view the plot.
    :param bool return_figure: 
        Option to return the figure and axes instances of the ``matplotlib`` classes.

    """
    if sample_data is None:
        X = Polynomial.get_points()
        y_truth = Polynomial._model_evaluations
        y_model = Polynomial.get_polyfit(X)
    else:
        X, y_truth = sample_data[0], sample_data[1]
        y_model = Polynomial.get_polyfit(X)
    R2score = score(y_truth, y_model, metric, X)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    plt.plot(y_model, y_truth, 'o', color='dodgerblue', ms=10, markeredgecolor='k',lw=1, alpha=0.6)
    plt.xlabel('Polynomial model')
    plt.ylabel('True data')
    displaytext = '$R^2$ = '+str(np.round(float(R2score), 2))
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])
    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    plt.text(0.3, 0.9, displaytext, transform=ax.transAxes, \
        horizontalalignment='center', verticalalignment='center', fontsize=14, color='grey')
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('model_vs_data_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig, ax
