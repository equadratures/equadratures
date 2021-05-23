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
    """ Generates a sufficient summary plot for 1D or 2D polynomial ridge approximations.

    Parameters
    ----------
    mysubspace : Subspaces
        An instance of the Subspaces class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    X_test : numpy.ndarray, optional
        A numpy ndarray containing input test data to plot (in addition to the training data). Must have same number of dimensions as the training data. 
    y_test : numpy.ndarray, optional
        A numpy ndarray containing output test data to plot (in addition to the training data).
    show : bool, optional
        Option to view the plot.
    poly : bool, optional
        Option to plot the subspace polynomial.
    uncertainty : bool, optional
        Option to show confidence intervals (1 standard deviation) of the subspace polynomial.
    legend : bool, optional
        Option to show legend.
    scatter_kwargs : dict, optional
        Dictionary of keyword arguments to pass to matplotlib.scatter().  
    plot_kwargs : dict, optional
        Dictionary of keyword arguments to pass to matplotlib.plot().  

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.

    Example
    -------
    >>> mysubspace = Subspaces(method='active-subspace', sample_points=X, sample_outputs=Y)
    >>> fig, ax = mysubspace.plot_sufficient_summary()
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
    else:
        fig = ax.figure

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
    return fig, ax

def plot_2D_contour_zonotope(mysubspace, minmax=[-3.5, 3.5], grid_pts=180,  \
                             show=True, ax=None):
    """ Generates a 2D contour plot of the polynomial ridge approximation.
    
    Parameters
    ----------
    mysubspace : Subspaces
        An instance of the Subspaces class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    minmax : list, optional
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    grid_pts : int, optional
        The number of grid points for generating the contour plot.
    show : bool, optional 
        Option to view the plot.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.

    Example
    -------
    >>> mysubspace = Subspaces(method='active-subspace', sample_points=X, sample_outputs=Y)
    >>> fig, ax = mysubspace.plot_2D_contour_zonotope()
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
    """ Generates a zonotope plot where samples from the second subspace are projected
    over the first.
    
    Parameters
    ----------
    mysubspace_1 : Subspaces
        An instance of the Subspaces class, to project samples onto.
    mysubspace_2 : Subspaces
        A second instance of the Subspaces class, to generate samples from.
    axs : list, optional
        A len(2) list containing two instances of the ``matplotlib`` axes class to plot onto. If ``None``, new figures and axes are created (default: ``None``).
    no_of_samples : int, optional
        Number of inactive samples to be generated.
    minmax : list, optional
        A list of the minimum and maximum values of :math:`M^T x`, where :math:`M` is the subspace.
    grid_pts : int, optional
        The number of grid points for generating the contour plot.
    show : bool, optional
        Option to view the plot.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`, :obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes` ) containing contour plots of the sample mean and variance over the 2D subspace.

    Example
    -------
    >>> mysubspace1 = Subspaces(method='active-subspace', sample_points=X, sample_outputs=Y)
    >>> mysubspace2 = Subspaces(method='variable-projection', sample_points=X, sample_outputs=Y)
    >>> fig1, ax1, fig2, ax2 = mysubspace1.plot_samples_from_second_subspace_over_first(mysubspace2)
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
    """ Plots a polynomial's Sobol' indices of a given order.

    Parameters
    ----------
    Polynomial : Poly 
        An instance of the Poly class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    order : int, optional
        Order of the Sobol' indices to plot.
    parameters : list, optional
        List of the parameters for the given polynomial.
    show : bool, optional
        Option to show the graph.
    kwargs : dict, optional
        Dictionary of keyword arguments to pass to matplotlib.bar().  

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
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
    sns.despine(offset=10, trim=True)

    if show:
        plt.show()
    return fig, ax

def plot_total_sobol(Polynomial, ax=None, show=True, labels=None, kwargs={}):
    """ Plots a polynomial's total-order Sobol' indices.

    Parameters
    ----------
    Polynomial : Poly 
        An instance of the Poly class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    parameters : list, optional 
        List of the parameters for the given polynomial.
    show : bool, optional 
        Option to show the graph.
    kwargs : dict, optional
        Dictionary of keyword arguments to pass to matplotlib.bar().  

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
    """
    # Set default kwargs
    kwargs = set_defaults(kwargs, {'color':'dodgerblue', 'ec':'k','lw':2, 'alpha':0.7})

    if ax is None:
        fig,ax = plt.subplots(figsize=(9, 6),tight_layout=True)
    else:
        ax.figure

    ndims = Polynomial.dimensions
    sobol_indices=Polynomial.get_total_sobol_indices()

    ax.set_ylabel(r'$S_{T_i}$')
    if labels is None: labels=[r'$S_{T_%d}$' %i for i in range(ndims)]
    to_plot = [sobol_indices[(i,)] for i in range(ndims)]

    ax.set_xticks(np.arange(ndims))
    ax.set_xticklabels(labels,rotation=45,rotation_mode="anchor",ha='right')
    ax.bar(np.arange(ndims),to_plot,**kwargs)
    sns.despine(offset=10, trim=True)

    if show:
        plt.show()
    return fig, ax
    
def plot_sobol_heatmap(Polynomial,parameters=None,show=True,ax=None):
    """ Generates a heatmap showing the first and second order Sobol indices. 

    Parameters
    ----------
    Polynomial : Poly 
          An instance of the Poly class.
    parameters : list 
          A list of strings to use for the axis labels.
    ax : matplotlib.axes.Axes 
          An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    show : bool
          Option to show the graph.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(10, 7),tight_layout=True)
        ax.set_xlabel(r'$u_1$')
        ax.set_ylabel(r'$u_2$')
    else:
        fig = ax.figure
    diag=Polynomial.get_sobol_indices(order=1)
    vals=Polynomial.get_sobol_indices(order=2)
    arr=np.ones(shape=(Polynomial.dimensions,Polynomial.dimensions))
    for val in vals.keys():
        arr[val[0]][val[1]]=vals[val]
    for val in diag.keys():
        arr[val[0]][val[0]]=diag[val]
    row=len(arr)
    col=len(arr[0])
    for i in range(row):
        for j in range(col):
            if j<i: arr[i][j]=float("NaN")
    cmap=sns.color_palette("Blues", as_cmap=True)
    if parameters is None:
        sns.heatmap(arr,annot=True,cmap=cmap,cbar_kws={'label':'Sobol Indices'})
    else:
        ax=sns.heatmap(arr,annot=True,xticklabels=parameters,yticklabels=parameters,cmap=cmap,cbar_kws={'label':'Sobol Indices'})
        ax.set_xticklabels(ax.get_xticklabels(),rotation=0 ,FontSize=10) 
        ax.set_yticklabels(ax.get_yticklabels(),rotation=0 ,FontSize=10) 
    if show:
        plt.show()
    return fig, ax

def plot_regpath(solver,elements=None,nplot=None,show=True):
    """ Generates the regularisation path for the :class:`~equadratures.solver.elastic_net` solver.

    Parameters
    ----------
    solver : Solver
        An instance of the Solver class.
    elements : numpy.ndarray, optional
        Elements of the index set to label the coefficients with. Typically set internally when this function is called by :meth:`~equadratures.solver.elastic_net.plot_regpath`.
    nplot : int, optional
        Number of coefficients for the plot.
    show : bool, optional
        Option to show the graph.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`, :obj:`~matplotlib.axes.Axes`) containing figure and two axes corresponding to the polynomial coefficients and information criterion plots.
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
        if elements is None:
            for j in plots:
                label="j=%d"%j
                ax1.plot(lamdas,x_path[:,j],'-',label=label,lw=2)
            
        else:
            for j in plots:
                e1 = elements[j,0]
                e2 = elements[j,1]
                if e1 == 0:
                    label = r'$p_%d(x_2)$' %e2
                elif e2 == 0:
                    label = r'$p_%d(x_1)$' %e1
                else:
                    label = r'$p_%d(x_1)p_%d(x_2)$' %(e1,e2)
                ax1.plot(lamdas,x_path[:,j],'-',label=label,lw=2)

        ax1.vlines(lamdas[idx],ax1.get_ylim()[0],ax1.get_ylim()[1],'k',ls='--')
        fig.legend(loc='center left', bbox_to_anchor=(1, 0.6),ncol=1,edgecolor='0.0')
        ax2.grid(True)
        ax2.set_xlabel('Log($\\lambda$)')
        ax2.set_ylabel(solver.crit)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.plot(lamdas,IC,'-k',lw=2)
        if IC_std is not None: 
            plt.fill_between(lamdas,IC-IC_std,IC+IC_std,alpha=0.3)
        ax2.vlines(lamdas[idx],ax2.get_ylim()[0],ax2.get_ylim()[1],'k',ls='--')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        sns.despine(fig=fig, offset=10, trim=False)
        if show:
            plt.show()
        return fig,ax1,ax2

def plot_pdf(Parameter, ax=None, data=None, show=True):
    """ Plots the probability density function for a Parameter.

    Parameters
    ----------
    Parameter : Parameter
        An instance of the Parameter class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    data : numpy.ndarray, optional
        Samples from the distribution (or a similar one) that need to be plotted as a histogram.
    show : bool, optional
        Option to show the graph.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
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
    sns.despine(ax=ax, offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def _trim_axs(ax,N):
    """ Private function to reduce *axs* to *N* axes. All further axes are removed from the figure.

    Parameters
    ----------
    axs : matplotlib.axes.Axes
        An instance of the ``matplotlib`` axes class to plot onto. 
    N : int 
        The number of axes to reduce the subplot to.

    Returns
    -------
    matplotlib.axes.Axes
        An axes, reduced to size *N*.
    """
    axs = axs.flat
    for a in axs[N:]:
        a.remove()
    return axs[:N]

def plot_parameters(Polynomial, ax=None, cols=2, show=True):
    """ Plots the probability density functions for all Parameters within a Polynomial.

    Parameters
    ----------
    Polynomial : Poly
        An instance of the Poly class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    cols : int, optional
        The number of columns to organise the parameter PDF plots into.
    show : bool, optional
        Option to show the graph.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
    """
    rows = len(Polynomial.parameters) // cols + 1
    if ax is None:
        fig, ax = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), tight_layout=True)
        ax = _trim_axs(ax, len(Polynomial.parameters))
    else:
        fig = ax.figure  
    for a, param in zip(ax, Polynomial.parameters):
        plot_pdf(param, ax=a, show=False)
        a.set_xlabel(param.variable.capitalize())
        a.set_ylabel('PDF')
    if show:
        plt.show()
    return fig, ax

def plot_orthogonal_polynomials(Parameter, ax=None, order_limit=None, number_of_points=200, show=True):
    """ Plots the first few orthogonal polynomials.

    Parameters
    ----------
    Parameter : Parameter
        An instance of the Parameter class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    order_limit : int, optional
        The maximum number of orthogonal polynomials that need to be plotted.
    number_of_points : int, optional
        The number of points used for plotting.
    show : bool, optional
        Option to view the plot.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.

    Example
    -------
        >>> myparam = eq.Parameter(distribution='uniform', lower = -1.0, upper = 1.0, order=8, endpoints='both')
        >>> myparam.plot_orthogonal_polynomials()        
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
    """ Plots a 1D only polynomial fit to the data.

    Parameters
    ----------
    Polynomial : Poly
        An instance of the Poly class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    uncertainty : bool, optional
        Option to show confidence intervals (1 standard deviation).
    output_variances : numpy.ndarray, optional
        User-defined uncertainty associated with each data point; can be either a ``float`` in which case all data points are assumed to have the same variance, or can be an array of length equivalent to the number of data points.
    show : bool, optional
        Option to view the plot.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
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
    """ Plots the polynomial approximation against the true data.

    Parameters
    ----------
    Polynomial : Poly 
        An instance of the Poly class.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    sample_data : list, optional
        A list formed by ``[X, y]`` where ``X`` represents the spatial data input and ``y`` the output.
    metric : str, optional
        Accuracy/error score metric to annotate graph with. See :meth:`~equadratures.datasets.score` for options.
    show : bool , optional
        Option to view the plot.

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`) containing the generated figure and axes.
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
    myscore = score(y_truth, y_model, metric, X)
    ax.plot(y_model, y_truth, 'o', color='dodgerblue', ms=10, markeredgecolor='k',lw=1, alpha=0.6)
    displaytext = '$Score$ = '+str(np.round(float(myscore), 2))
    ax.text(0.3, 0.9, displaytext, transform=ax.transAxes, \
        horizontalalignment='center', verticalalignment='center', fontsize=14, color='grey')
    sns.despine(offset=10, trim=True)
    if show:
        plt.show()
    return fig, ax

def plot_decision_surface(PolyTree,ij,ax=None,X=None,y=None,max_depth=None,label=True,
                                 color='data',colorbar=True,show=True,kwargs={}):
    """ Plots the decision boundaries of the PolyTree over a 2D surface.

    Parameters
    ----------
    PolyTree : PolyTree
        An instance of the PolyTree class.
    ij : list, optional
        A list containing the two dimensions to plot over. For example, ``ij=[6,7]`` with plot over the 6th and 7th dimensions in ``X``.
    ax : matplotlib.axes.Axes, optional
        An instance of the ``matplotlib`` axes class to plot onto. If ``None``, a new figure and axes are created (default: ``None``).
    X : numpy.ndarray, optional
        A numpy ndarray containing the input data to plot.
    y : numpy.ndarray, optional
        A numpy ndarray containing the output data to plot.
    max_depth : int, optional
        The maximum tree depth to plot decision boundaries for.
    label : bool, optional
        If ``True`` then the subdomains are labelled by their node number.
    color : str, optional
        What to color the scatter points by. ``'data'`` to color by the **X**, **y** data. ``'predict'`` to color by the PolyTree predictions, and ``'error'`` to color by the predictive error. (default: ``'data'``).
    colorbar : bool, optional
        Option to add a colorbar.
    show : bool, optional
        Option to view the plot.
    kwargs : dict, optional
        Dictionary of keyword arguments to pass to matplotlib.scatter().  

    Returns
    -------
    tuple
        Tuple (:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`, :obj:`~matplotlib.collections.PathCollection`) containing the figure, axes and handle for the scatter plot.
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
            col_label = 'Error'
    elif color.lower() == 'predict':
            scat = ax.scatter(X[:,ij[0]],X[:,ij[1]],c=PolyTree.predict(X),**kwargs)
            col_label = 'Prediction'
    elif color.lower() == 'data':
            scat = ax.scatter(X[:,ij[0]],X[:,ij[1]],c=y,**kwargs)
            col_label = 'Truth'
    else:
        raise ValueError("color argument should be set to 'error', 'predict', or 'data'")
    if colorbar:
        fig.colorbar(scat, orientation="vertical", pad=0.1, shrink=0.5,label=col_label)

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
                                       node["Xmax"][1],'k',lw=3)
                    else:
                            ax.hlines(split_vals[n],node["Xmin"][0],
                                       node["Xmax"][0],'k',lw=3)

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
