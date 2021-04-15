"""Plotting utilities."""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from equadratures.datasets import score
import numpy as np
# seaborn defaults
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_style("ticks")
palette = sns.color_palette('deep')

# matplotlib defaults
mpl.rcParams['figure.facecolor'] = 'none'

def plot_sufficient_summary(mysubspace, ax=None, X_test=None, y_test=None, show=True, poly=True, uncertainty=False, legend=False, scatter_kwargs={}, plot_kwargs={}):
    """
    Generates a sufficient summary plot for 1D or 2D polynomial ridge approximations.
    
    :param Subspace mysubspace:
        An instance of the Subspace class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param :numpy.ndarray X_test:
        A numpy ndarray containing input test data to plot (in addition to the training data). Must have same number of dimensions as the training data. 
    :param :numpy.ndarray y_test:
        A numpy ndarray containing output test data to plot (in addition to the training data).
    :param bool show:
        Option to view the plot.
    :param bool poly:
        Option to plot the subspace polynomial.
    :param bool uncertainty:
        Option to show confidence intervals (1 standard deviation) of the subspace polynomial.
    :param bool legend:
        Option to show legend.
    :param dict scatter_kwargs:
        Dictionary of keyword arguments to pass to matplotlib.scatter().  
    :param dict plot_kwargs:
        Dictionary of keyword arguments to pass to matplotlib.plot().  
    """
    # Set default kwargs
    scatter_kwargs = set_defaults(scatter_kwargs,{'marker':'o','s':60,'ec':'k','lw':2,'alpha':0.7})
    plot_kwargs    = set_defaults(plot_kwargs   ,{'lw':3})

    # 1D or 2D ridge
    subdim = mysubspace.subspace_dimension

    if ax is None:
        if subdim == 1:
            fig,ax = plt.subplots(figsize=(7, 5),tight_layout=True)
            ax.set_xlabel(r'$u$')
            ax.set_ylabel(r'$y$')
        elif subdim == 2:
            fig = plt.figure(figsize=(9,7),tight_layout=True)
            ax = fig.add_subplot(111, projection='3d')
            ax.zaxis.set_rotate_label(False)
            ax.set_xlabel(r'$u_1$',labelpad=10)
            ax.set_ylabel(r'$u_2$',labelpad=10)
            ax.set_zlabel(r'$y$'  ,labelpad=10)
        else:
            raise ValueError("Currently can only generate sufficient summary plot for 1D and 2D ridge approximations")

    X = mysubspace.sample_points
    y = mysubspace.sample_outputs
    M = mysubspace.get_subspace()
    W = M[:,:subdim]
    u = X @ W
    if (X_test is not None) and (y_test is not None):
        u_test = X_test @ W
        test = True
    else:
        test = False
    if poly:
        subpoly = mysubspace.get_subspace_polynomial()

    if subdim == 1:
        ax.scatter(u, y, color=palette[0], label='Training samples', **scatter_kwargs)
        if test:
            ax.scatter(u_test, y_test, color=palette[2], label='Test samples', **scatter_kwargs)
        if poly:
            u_samples = np.linspace(np.min(u[:,0]), np.max(u[:,0]), 50)
            if uncertainty:
                y_mean, y_std = subpoly.get_polyfit(u_samples,uq=True)
                ax.fill_between(u_samples,(y_mean-y_std).squeeze(),(y_mean+y_std).squeeze(), color=palette[3], alpha=0.3, label=r'Polynomial  $\pm\sigma$')
            else:
                y_mean = subpoly.get_polyfit(u_samples)
            ax.plot(u_samples,y_mean,color=palette[3],label='Polynomial approx.',**plot_kwargs)

    elif subdim == 2:
        ax.scatter(u[:,0], u[:,1], y, color=palette[0], label='Training samples',**scatter_kwargs)
        if test:
            ax.scatter(u_test[:,0], u_test[:,1], y_test, color=palette[2], label='Test samples')
        if poly:
            subpoly = mysubspace.get_subspace_polynomial()
            N = 40
            u1_samples = np.linspace(np.min(u[:,0]), np.max(u[:,0]), N)
            u2_samples = np.linspace(np.min(u[:,1]), np.max(u[:,1]), N)
            [U1, U2] = np.meshgrid(u1_samples, u2_samples)
            u_samples = np.hstack([U1.reshape(N*N,1), U2.reshape(N*N,1)])
            if uncertainty:
                y_mean, y_std = subpoly.get_polyfit(u_samples, uq=True)
                y_mean = y_mean.reshape(N,N)
                y_std  = y_std.reshape(N,N)
                ax.plot_surface(U1, U2, y_mean-y_std, rstride=2, cstride=2, color=palette[3], ec='None', alpha=0.2, label='_nolegend_')
                ax.plot_surface(U1, U2, y_mean+y_std, rstride=2, cstride=2, color=palette[3], ec='None', alpha=0.2, label='_nolegend_')
            else:
                y_mean = subpoly.get_polyfit(u_samples).reshape(N,N)
            ax.plot_surface(U1, U2, y_mean, rstride=2, cstride=2, color=palette[3], ec='None', alpha=0.7, label='_nolegend_')

    sns.despine(offset=10, trim=True)

    if legend:
        ax.legend(ncol=2)
    if show:
        plt.show()
    if 'fig' in locals():
        return fig, ax
    else:
        return ax


def plot_2D_contour_zonotope(mysubspace, minmax=[-3.5, 3.5], grid_pts=180,  \
                             show=True, ax=None):
    """
    Generates a 2D contour plot of the polynomial ridge approximation.
    
    :param Subspace mysubspace:
        An instance of the Subspace class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param list minmax:
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    :param int grid_pts:
        The number of grid points for generating the contour plot.
    :param bool show: 
        Option to view the plot.
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(10, 7),tight_layout=True)
        ax.set_xlabel(r'$u_1$')
        ax.set_ylabel(r'$u_2$')
    else:
        fig = ax.figure
   
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
    norm = mpl.colors.Normalize(vmin=np.min(mysubspace.sample_outputs) - 0.4 * np.std(mysubspace.sample_outputs),\
                                               vmax=np.max(mysubspace.sample_outputs)+ 0.4 * np.std(mysubspace.sample_outputs))
    ax.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False, norm=norm)  
    c = ax.scatter(u[:,0],u[:,1], c=mysubspace.sample_outputs, marker='o', edgecolor='w', lw=1, alpha=0.8, s=80, norm=norm )  
    fig.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5,label='y')
    for simplex in hull.simplices:
        ax.plot(pts[simplex, 0], pts[simplex, 1], 'k-', lw=5)
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax
    
def plot_samples_from_second_subspace_over_first(mysubspace_1, mysubspace_2, axs=None, no_of_samples=500,\
                             minmax=[-3.5, 3.5], grid_pts=180, show=True): 
    """
    Generates a zonotope plot where samples from the second subspace are projected
    over the first.
    
    
    :param Subspace mysubspace_1:
        An instance of the Subspace class.
    :param Subspace mysubspace_2:
        A second instance of the Subspace class.
    :param list axs: 
        A len(2) list containing two instances of the ``matplotlib`` axes class to plot onto. If ``None``, new figures and axes are created (default: ``None``).
    :param int no_of_samples:
        Number of inactive samples to be generated.
    :param list minmax:
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    :param int grid_pts:
        The number of grid points for generating the contour plot.
    :param bool show: 
        Option to view the plot.
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

    if axs is None:
        fig1,ax1 = plt.subplots(figsize=(10, 7),tight_layout=True)
        fig2,ax2 = plt.subplots(figsize=(10, 7),tight_layout=True)
        ax1.set_xlabel(r'$u_1$')
        ax1.set_ylabel(r'$u_2$')
        ax2.set_xlabel(r'$u_1$')
        ax2.set_ylabel(r'$u_2$')
    else:
        ax1 = axs[0]
        ax2 = axs[1]
        fig1 = ax1.figure
        fig2 = ax2.figure

    # Contour plot of mean values
    subspace_poly_evals[indices] =  mean_values
    c = ax1.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False)  
    fig1.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5,label=r'$\mu$')
    for simplex in hull.simplices:
        ax1.plot(pts_zono[simplex, 0], pts_zono[simplex, 1], 'k-', lw=5)

    # Contour plot of std values
    subspace_poly_evals[indices] =  std_values
    c = ax2.contourf(XX, YY, subspace_poly_evals.reshape(grid_pts,grid_pts), levels=40, linewidth=0, \
                antialiased=False)  
    fig2.colorbar(c, orientation="vertical", pad=0.1, shrink=0.5,label=r'$\sigma$')
    for simplex in hull.simplices:
        ax2.plot(pts_zono[simplex, 0], pts_zono[simplex, 1], 'k-', lw=5)

    sns.despine(fig1,offset=10, trim=True)
    sns.despine(fig2,offset=10, trim=True)
    if show:
        plt.show()
    return fig1, ax1, fig2, ax2

def plot_sobol(Polynomial, ax=None, order=1, show=True, labels=None, kwargs={}):
    """
    Plots a polynomial's Sobol' indices of a given order.

    :param Poly Polynomial: 
        An instance of the Poly class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param int order:
        Order of the Sobol' indices to plot.
    :param list parameters: 
        List of the parameters for the given polynomial.
    :param bool show: 
        Option to show the graph.
    :param dict kwargs:
        Dictionary of keyword arguments to pass to matplotlib.bar().  
    """
    # Set default kwargs
    kwargs = set_defaults(kwargs, {'color':'dodgerblue', 'ec':'k','lw':2, 'alpha':0.7})

    if ax is None:
        fig,ax = plt.subplots(figsize=(9, 6),tight_layout=True)
    else:
        fig = ax.figure

    ndims = Polynomial.dimensions
    sobol_indices=Polynomial.get_sobol_indices(order)
    nsob = len(sobol_indices)
    if (order > 3):
        raise ValueError("Only Sobol' indices of order<=3 can currently be plotted")
    if (nsob == 0):
        raise ValueError("Insufficient number of parameters to obtain order=%d Sobol' indices" %order)

    if order == 1:
        ax.set_ylabel(r'$S_i$')
        if labels is None: labels=[r'$S_%d$' %i for i in range(ndims)]
        to_plot = [sobol_indices[(i,)] for i in range(ndims)]

    elif order == 2:
        ax.set_ylabel(r'$S_{ij}$')
        if labels is None:
            labels=[r'$S_{%d%d}$' %(i,j) for i in range(ndims) for j in range(i+1,ndims)]
        else:
            labels=[labels[i] + r' $\leftrightarrow$ ' + labels[j]  for i in range(ndims) for j in range(i+1,ndims)]
        to_plot = [sobol_indices[(i,j)] for i in range(ndims) for j in range(i+1,ndims)]

    elif order == 3:
        ax.set_ylabel(r'$S_{ijk}$')
        if labels is None:
            labels=[r'$S_{%d%d%d}$' %(i,j,k) for i in range(ndims) for j in range(i+1,ndims) for k in range(j+1,ndims)]
        else:
            labels=[labels[i] + r' $\leftrightarrow$ ' + labels[j] + r' $\leftrightarrow$ ' + labels[k]
                    for i in range(ndims) for j in range(i+1,ndims) for k in range(j+1,ndims)]
        to_plot = [sobol_indices[(i,j,k)] for i in range(ndims) for j in range(i+1,ndims) for k in range(j+1,ndims)]

    ax.set_xticks(np.arange(nsob))
    ax.set_xticklabels(labels,rotation=45,rotation_mode="anchor",ha='right')
    ax.bar(np.arange(nsob),to_plot,**kwargs)
    sns.despine(fig)

    if show:
        plt.show()
    return fig, ax

def plot_total_sobol(Polynomial, ax=None, show=True, labels=None, kwargs={}):
    """
    Plots a polynomial's total-order Sobol' indices.

    :param Poly Polynomial: 
        An instance of the Poly class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param list parameters: 
        List of the parameters for the given polynomial.
    :param bool show: 
        Option to show the graph.
    :param dict kwargs:
        Dictionary of keyword arguments to pass to matplotlib.bar().  
    """
    # Set default kwargs
    kwargs = set_defaults(kwargs, {'color':'dodgerblue', 'ec':'k','lw':2, 'alpha':0.7})

    if ax is None:
        fig,ax = plt.subplots(figsize=(9, 6),tight_layout=True)

    ndims = Polynomial.dimensions
    sobol_indices=Polynomial.get_total_sobol_indices()

    ax.set_ylabel(r'$S_{T_i}$')
    if labels is None: labels=[r'$S_{T_%d}$' %i for i in range(ndims)]
    to_plot = [sobol_indices[(i,)] for i in range(ndims)]

    ax.set_xticks(np.arange(ndims))
    ax.set_xticklabels(labels,rotation=45,rotation_mode="anchor",ha='right')
    ax.bar(np.arange(ndims),to_plot,**kwargs)
    sns.despine(fig)

    if show:
        plt.show()
    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_regpath(solver,nplot=None,save=False,show=True,return_figure=False):
    """
    Generates a regularisation path for elastic net.

    :param Poly Polynomial: 
        An instance of the Poly class.
    :param int nplot:
        Number of coefficients for the plot.
    :param bool save: 
        Option to save the plot as a .png file.
    :param bool show: 
        Option to show the graph.
    :param bool return_figure: 
        Option to get the figure axes,figure.

    """
    lamdas = solver.lambdas
    x_path = solver.xpath
    IC = solver.ic
    IC_std = solver.ic_std
    idx = solver.opt_idx
    if nplot is not None and nplot > x_path.shape[1]:
        raise ValueError("Max number of plots are {}".format(x_path.shape[1]))
    else:  
        fig, (ax1,ax2) = plt.subplots(figsize=(10,7),nrows=2,sharex=True,tight_layout=True)
        ax1.set_xscale('log')
        ax1.set_ylabel(r'$\theta$')
        ax1.grid(True)
        if nplot is None:
            plots = range(x_path.shape[1])
        else:
            coeffs = x_path[0,:]
            plots = (-np.abs(coeffs)).argsort()[:nplot]
        for j in plots:
            label="j=%d"%j       
            ax1.plot(lamdas,x_path[:,j],'-',label=label,lw=2)
        ax1.vlines(lamdas[idx],ax1.get_ylim()[0],ax1.get_ylim()[1],'k',ls='--')
        fig.legend(loc='center left', bbox_to_anchor=(1, 0.6),ncol=1,edgecolor='0.0')
        ax2.grid(True)
        ax2.set_xlabel('Log($\\lambda$)')
        ax2.set_ylabel('AIC')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.plot(lamdas,IC,'-k',lw=2)
        if IC_std is not None: 
            plt.fill_between(lamdas,IC-IC_std,IC+IC_std,alpha=0.3)
        ax2.vlines(lamdas[idx],ax2.get_ylim()[0],ax2.get_ylim()[1],'k',ls='--')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        if save:
            plt.savefig('regularisation_path.png', dpi=140, bbox_inches='tight')
        if show:
            plt.show()
        if return_figure:
            return fig,(ax1,ax2)

def plot_pdf(Parameter, ax=None, data=None, show=True):
    """
    Plots the probability density function for a Parameter.

    :param Parameter Parameter: 
        An instance of the Parameter class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param numpy.array data: 
        Samples from the distribution (or a similar one) that need to be plotted as a histogram.
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(8, 6),tight_layout=True)
        ax.set_xlabel(Parameter.variable.capitalize())
        ax.set_ylabel('PDF')
    else:
        fig = ax.figure
    s_values, pdf = Parameter.get_pdf()
    ax.fill_between(s_values,  pdf*0.0, pdf, color="gold" , label='Density', interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
    if data is not None:
        ax.hist(data, 50, density=True, facecolor='dodgerblue', alpha=0.7, label='Data', edgecolor='white')
    ax.legend()
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def plot_orthogonal_polynomials(Parameter, ax=None, order_limit=None, number_of_points=200, show=True):
    """
    Plots the first few orthogonal polynomials.

    :param Parameter Parameter: 
        An instance of the Parameter class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param int order_limit:
        The maximum number of orthogonal polynomials that need to be plotted.
    :param int number_of_points: 
        The number of points used for plotting.
    :param bool show: 
        Option to view the plot.

    **Example**::

        import numpy as np
        from equadratures import *

        myparam = eq.Parameter(distribution='uniform', lower = -1.0, upper = 1.0, order=8, endpoints='both')
        myparam.plot_orthogonal_polynomials()
        
    """
    Xi = np.linspace(Parameter.distribution.x_range_for_pdf[0], \
                Parameter.distribution.x_range_for_pdf[-1], number_of_points).reshape(number_of_points, 1)
    P, _, _ = Parameter._get_orthogonal_polynomial(Xi)
    if ax is None:
        fig,ax = plt.subplots(figsize=(8, 6),tight_layout=True)
        ax.set_xlabel(Parameter.name.capitalize()+' parameter')
        ax.set_ylabel('Orthogonal polynomials')
    else:
        fig = ax.figure
    if order_limit is None:
        max_order = P.shape[0]
    else:
        max_order = order_limit
    for i in range(0, max_order):
        ax.plot(Xi, P[i,:], '-', lw=2, label='order %d'%(i))
    ax.legend()
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def plot_polyfit_1D(Polynomial, ax=None, uncertainty=True, output_variances=None, number_of_points=200, show=True):
    """
    Plots a 1D only polynomial fit to the data.

    :param Poly Polynomial: 
        An instance of the Polynomial class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param bool uncertainty:
        Option to show confidence intervals (1 standard deviation).
    :param numpy.array output_variances:
        User-defined uncertainty associated with each data point; can be either a ``float`` in which case all data points are assumed to have the same variance, or can be an array of length equivalent to the number of data points.
    :param bool show: 
        Option to view the plot.
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(8, 6),tight_layout=True)
        ax.set_xlabel(Polynomial.parameters[0].variable.capitalize())
        ax.set_ylabel('Polynomial fit')
    else:
        fig = ax.figure
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
    ax.plot(Xi, y, '-',label='Polynomial fit', color='navy')
    ax.plot(X.flatten(), y_truth.flatten(), 'o', color='dodgerblue', ms=10, markeredgecolor='k',lw=1, alpha=0.6, label='Data')
    if uncertainty:
        ax.fill_between(Xi.flatten(), y+ystd, y-ystd, alpha=.10, color='deepskyblue',label='Polynomial $\sigma$')
    ax.legend()
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def plot_model_vs_data(Polynomial, ax=None, sample_data=None, metric='adjusted_r2', show=True):
    """
    Plots the polynomial approximation against the true data.

    :param Polynomial Polynomial: 
        An instance of the Poly class.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param list sample_data:
        A list formed by ``[X, y]`` where ``X`` represents the spatial data input and ``y`` the output.
    :param bool show: 
        Option to view the plot.
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(8, 6),tight_layout=True)
        ax.set_xlabel('Polynomial model')
        ax.set_ylabel('True data')
    else:
        fig = ax.figure
    if sample_data is None:
        X = Polynomial.get_points()
        y_truth = Polynomial._model_evaluations
        y_model = Polynomial.get_polyfit(X)
    else:
        X, y_truth = sample_data[0], sample_data[1]
        y_model = Polynomial.get_polyfit(X)
    score = score(y_truth, y_model, metric, X)
    ax.plot(y_model, y_truth, 'o', color='dodgerblue', ms=10, markeredgecolor='k',lw=1, alpha=0.6)
    displaytext = '$Score$ = '+str(np.round(float(score), 2))
    ax.text(0.3, 0.9, displaytext, transform=ax.transAxes, \
        horizontalalignment='center', verticalalignment='center', fontsize=14, color='grey')
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def plot_decision_surface(PolyTree,ij,ax=None,X=None,y=None,max_depth=None,label=True,
                                 color='data',colorbar=True,show=True,kwargs={}):
    """
    Plots the decision boundaries of the PolyTree over a 2D surface.

    :param PolyTree PolyTree: 
        An instance of the PolyTree class.
    :param list ij: 
        A list containing the two dimensions to plot over. For example, ``ij=[6,7]`` with plot over the 6th and 7th dimensions in ``X``.
    :param matplotlib.ax ax: 
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    :param :numpy.ndarray X:
        A numpy ndarray containing the input data to plot.
    :param :numpy.ndarray y:
        A numpy ndarray containing the output data to plot.
    :param int max_depth:
        The maximum tree depth to plot decision boundaries for.
    :param bool label:
        If ``True`` then the subdomains are labelled by their node number.
    :param string color:
        What to color the scatter points by. ``'data'`` to color by the ``X``,``y`` data. ``'predict'`` to color by the PolyTree predictions, and ``'error'`` to color by the predictive error. (default: ``'data'``).
    :param bool colorbar:
        Option to add a colorbar.
    :param bool show:
        Option to view the plot.
    :param dict kwargs:
        Dictionary of keyword arguments to pass to matplotlib.scatter().  
    """
    # Set default kwargs
    kwargs = set_defaults(kwargs, {'alpha':0.8,'ec':'lightgray','cmap':'coolwarm'})

    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.figure

    # TODO - What to do in case where "data" doesn't exist? Have domain\subdomain attribute for each node?
    if X is None:
            X = PolyTree.tree["data"][0]
    if y is None:
            y = PolyTree.tree["data"][1]
    Xij = X[:,ij]
    PolyTree.tree["Xmin"] = np.min(Xij,axis=0)
    PolyTree.tree["Xmax"] = np.max(Xij,axis=0)

    if color.lower() == 'error':
            error = y - PolyTree.predict(X)
            scat = ax.scatter(X[:,ij[0]],X[:,ij[1]],c=error,**kwargs)
            label = 'Error'
    elif color.lower() == 'predict':
            scat = ax.scatter(X[:,ij[0]],X[:,ij[1]],c=PolyTree.predict(X),**kwargs)
            label = 'Prediction'
    elif color.lower() == 'data':
            scat = ax.scatter(X[:,ij[0]],X[:,ij[1]],c=y,**kwargs)
            label = 'Truth'
    else:
        raise ValueError("color argument should be set to 'error', 'predict', or 'data'")
    if colorbar:
        fig.colorbar(scat, orientation="vertical", pad=0.1, shrink=0.5,label=label)

    def _get_boundaries(nodes,final):
            # Find leaf nodes
            left_children = [node["children"]["left"] for node in nodes]
            leaf_nodes = np.array([True if node is None else False for node in left_children])

            # Get splitting info from non-leaf nodes (i.e. split nodes)
            split_nodes = nodes[~leaf_nodes]
            split_dims = [node["j_feature"] for node in split_nodes]
            split_vals = [node["threshold"] for node in split_nodes]
            indices    = [node["index"]     for node in split_nodes]

            # Labelling done before splits, as we only label up to max_depth and then return
            if label:
                    # If final, label all nodes, else only leaf nodes
                    if final:
                            for node in nodes:
                                    ax.annotate('Node %d'%node["index"],(node["Xmax"][0],node["Xmax"][1]),
                                                ha='right',va='top',textcoords='offset points',
                                                xytext=(-5, -5))
                            return
                    else:
                            for node in nodes[leaf_nodes]:
                                    ax.annotate('Node %d'%node["index"],(node["Xmax"][0],node["Xmax"][1]),
                                                ha='right',va='top',textcoords='offset points',
                                                xytext=(-5, -5))

            # Plot split lines
            for n, node in enumerate(split_nodes):
                    if split_dims[n]==ij[0]:
                            ax.vlines(split_vals[n],node["Xmin"][1],
                                       node["Xmax"][1],'k')
                    else:
                            ax.hlines(split_vals[n],node["Xmin"][0],
                                       node["Xmax"][0],'k')

            # Update bounding boxes of child nodes before returning them
            for node in split_nodes:
                    if node["j_feature"]==ij[0]:
                            node["children"]["left"]["Xmax"]  = [node["threshold"],node["Xmax"][1]]
                            node["children"]["right"]["Xmin"] = [node["threshold"],node["Xmin"][1]]
                            node["children"]["left"]["Xmin"]  = node["Xmin"]
                            node["children"]["right"]["Xmax"] = node["Xmax"]
                    else:
                            node["children"]["left"]["Xmax"]  = [node["Xmax"][0],node["threshold"]]
                            node["children"]["right"]["Xmin"] = [node["Xmin"][0],node["threshold"]]
                            node["children"]["left"]["Xmin"]  = node["Xmin"]
                            node["children"]["right"]["Xmax"] = node["Xmax"]

            # Extract child node info for next level down
            left_nodes  = [node["children"]["left"]  for node in split_nodes]
            right_nodes = [node["children"]["right"] for node in split_nodes]

            child_nodes = np.array(left_nodes + right_nodes)

            return child_nodes

    nodes = np.array([PolyTree.tree])
    depth = 0
    final = False
    while len(nodes)>0:
            if depth==max_depth: final = True
            nodes = _get_boundaries(nodes,final)
            if final: break
            depth += 1

    if show:
        plt.show()
    return fig, ax, scat

def set_defaults(kwargs, defaults):
    for key in defaults:
        kwargs.setdefault(key, defaults[key])
    return kwargs
