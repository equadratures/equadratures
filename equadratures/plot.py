"""Plotting utilities."""
import seaborn as sns
import matplotlib.pyplot as plt
from equadratures.datasets import score
import numpy as np
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_style("ticks")
def plot_Sobol_indices(Polynomial, save=False, xlim=None, ylim=None, show=True, return_figure=False):
    """
    Generates a bar chart of the first order Sobol' indices.

    :param Poly Polynomial: 
        An instance of the Poly class.
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
    sobol = Polynomial.get_sobol_indices(1)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    data_1 = np.arange(7) - 0.10 + 1
    for i in range(0, len(sobol)):
        plt.bar(i+1, sobol[(i,)], color='steelblue',linewidth=1.5)
    plt.xlabel(r'Parameters', fontsize=16)
    plt.ylabel(r"First order Sobol' indices", fontsize=16)
    xTickMarks = [Polynomial.parameters[j].variable for j in range(0, Polynomial.dimensions)]
    ax.set_xticks(data_1+0.10)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=16)
    sns.despine(offset=10, trim=True)
    if save:
        plt.savefig('sobol_plot.png', dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    if return_figure:
        return fig, ax
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
