import seaborn as sns
import matplotlib.pyplot as plt
from equadratures.datasets import score
import numpy as np
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_style("ticks")

class Plot:
    """
    Plotting utilities.
    """
    def plot_pdf(self, data=None, save=False, xlim=None, ylim=None):
        """
        Plots the probability density function for a Parameter.
        """
        s_values, pdf = self.get_pdf()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        plt.fill_between(s_values,  pdf*0.0, pdf, color="gold" , label='Density', interpolate=True, hatch="\\\\\\\\", edgecolor="grey",  linewidth=0.5,alpha=0.5)
        if data is not None:
            plt.hist(data, 50, density=True, facecolor='dodgerblue', alpha=0.7, label='Data', edgecolor='white')
        plt.xlabel(self.variable.capitalize())
        plt.ylabel('PDF')
        if xlim is not None:
            plt.xlim([xlim[0], xlim[1]])
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])
        plt.legend()
        sns.despine(offset=10, trim=True)
        if save:
            plt.savefig('pdf_plot.png', dpi=140, bbox_inches='tight')
        else:
            plt.show()
    @staticmethod
    def plot_orthogonal_polynomials(Parameter, ax=None, order_limit=None, number_of_points=200, save=False, xlim=None, ylim=None, show=True):
        """
        Plots the first K orthogonal polynomials.

        :param Parameter Parameter: An instance of the Parameter class.
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
        return fig, ax
    def plot_polyfit_1D(self, uncertainty=True, output_variances=None, number_of_points=200, save=False, xlim=None, ylim=None):
        """
        Plots a univariate polynomial.
        """
        if self.dimensions != 1:
            raise(ValueError, 'plot_polyfit_1D is only meant for univariate polynomials.')
        Xi = np.linspace(self.parameters[0].distribution.x_range_for_pdf[0], \
                    self.parameters[0].distribution.x_range_for_pdf[-1], number_of_points).reshape(number_of_points, 1)
        if uncertainty:
            if output_variances is None:
                y, ystd = self.get_polyfit(Xi,uq=True)
            else:
                self.output_variances = output_variances
                y, ystd = self.get_polyfit(Xi,uq=True)
            ystd = ystd.squeeze()
        else:
            y = self.get_polyfit(Xi)
        y = y.squeeze()
        X = self.get_points()
        y_truth = self._model_evaluations
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
        plt.xlabel(self.parameters[0].variable.capitalize())
        plt.ylabel('Polynomial fit')
        if save:
            plt.savefig('polyfit_1D_plot.png', dpi=140, bbox_inches='tight')
        else:
            plt.show()

    def plot_model_vs_data(self, sample_data=None, metric='adjusted_r2', save=False, xlim=None, ylim=None):
        """
        Plots the polynomial approximation against the true data.

        :param Poly self: An instance of the Poly class.

        """
        if sample_data is None:
            X = self.get_points()
            y_truth = self._model_evaluations
            y_model = self.get_polyfit(X)
        else:
            X, y_truth = sample_data[0], sample_data[1]
            y_model = self.get_polyfit(X)
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
        else:
            plt.show()
