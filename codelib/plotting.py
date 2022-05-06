import numpy as np
from numpy import ndarray
from codelib.stats import weighted_percentile, weighted_skew, weighted_kurtosis, c_var
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cycler
import matplotlib as mpl
import scipy.stats as stats
from IPython.display import display, clear_output


class DefaultStyle:

    """
    Class the sets the defaults plotting style
    """

    def __init__(self, plot_size = (12, 6), font_size: float = 15.0):

        """
        Spills plot style into memory upon instantiation

        Parameters
        ----------
        plot_size:
        The size of the figure as (height, width)
        font_size:
        The size of the font on the plot as as float
        """

        self.colors = list(default_colors.values())
        self.figsize = plot_size
        self.font_size = font_size
        self.legend = True
        self._font = 'Arial'
        self._face_color = 'white'
        self._edge_color = 'white'
        self._grid_color = '#dddddd'
        self._tick_color = '.15'
        self._line_style = '--'
        self._line_width = 2.0
        self._tick_dir = 'out'
        self.save_format = 'pdf'
        self._spill()

    def _spill(self):

        """
        Spills rcParams into global memory

        Returns
        -------
        None
        """
        colors = cycler(color=self.colors)
        plt.rc('axes', facecolor=self._face_color, axisbelow=True, grid=True, prop_cycle=colors, autolimit_mode='data',
               xmargin=0, ymargin=0)
        plt.rc('grid', color=self._grid_color, linestyle=self._line_style)
        plt.rc('xtick', direction=self._tick_dir, color=self._tick_color)
        plt.rc('ytick', direction=self._tick_dir, color=self._tick_color)
        plt.rc('patch', edgecolor=self._edge_color)
        plt.rc('lines', linewidth=self._line_width)
        plt.rc('font', family='Arial', size=self.font_size)
        plt.rc('legend', loc='best', fontsize=self.font_size, fancybox=True, shadow=False)
        plt.rc('savefig', format=self.save_format)


def var_cvar_plot(x, probs=None, save_fig_title=False, color="blue", title=None, **kwargs):
    """
    Var vs cVar Plot
    """
    var = weighted_percentile(x, p=0.05, probs=probs, axis=0)
    cvar = c_var(x, p=0.05, probs=probs, axis=0)

    x_min = 0.25
    x_max = 3.5
    x_lim = (x_min, x_max)
    y_lim = (0., 3.75)
    initialize_fig = True
    var_cvar_loc_fs = [0.075, 0.95]

    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False
    if initialize_fig:
        fig, ax = plt.subplots()
    if "x_lim" in kwargs:
        x_lim = kwargs["x_lim"]
    if "y_lim" in kwargs:
        y_lim = kwargs["y_lim"]
    if "var_cvar_loc_fs" in kwargs:
        var_cvar_loc_fs = kwargs["var_cvar_loc_fs"]

    x_values, y_values = sns.kdeplot(x=x, color=color, alpha=0.05, weights=probs, ax=ax).get_lines()[0].get_data()
    sns.kdeplot(x=x, color=color, weights=probs, alpha=1, ax=ax)
    ax.axvline(x=var, color=color, linewidth=2, label="VaR")
    ax.axvline(x=cvar, ls="--", color=color, linewidth=2, label="CVaR")
    ax.fill_between(x_values, y_values, where=x_values < var, alpha=0.1, color=color)
    ax.text(var_cvar_loc_fs[0], var_cvar_loc_fs[1], f"  $VaR_{{0.05}}$ = {np.format_float_positional(var, 4)}",
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=var_cvar_loc_fs[2])
    ax.text(var_cvar_loc_fs[0], var_cvar_loc_fs[1] - 0.05, f"$CVaR_{{0.05}}$ = {np.format_float_positional(cvar, 4)}",
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=var_cvar_loc_fs[2])

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.set(xlim=x_lim, ylim=y_lim)

    if save_fig_title:
        fig.tight_layout()
        plt.savefig(f"plots/{save_fig_title}.png")


def volume_contribution_plot(time_points, volumes, save_fig_title=False, title=None, **kwargs):
    """
    Volume contribution plot
    :param time_points:
    :param volumes:
    :param save_fig_title:
    :param title:
    :param kwargs:
    """
    initialize_fig = True
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False
    if initialize_fig:
        fig, ax = plt.subplots()

    agents = ["Random", "Investor", "Trend", "MarketMaker"]

    ax.stackplot(time_points, volumes, labels=agents);
    #ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=6);
    ax.set_xlabel("Time")
    ax.set_ylabel("Percentage of Volume")
    ax.set_title(title);

    if save_fig_title:
        plt.tight_layout()
        plt.savefig(f'plots/{save_fig_title}.png')


def dist_vs_normal_plot(returns, **kwargs):
    """

    :param returns:
    :param kwargs:
    :return:
    """
    loc = [0.15, 0.9]
    avg = np.nanmean(returns)
    std = np.std(returns)[0]
    n = len(returns)
    x = np.linspace(np.max(returns)[0], np.min(returns)[0], n)
    norm_dist = stats.norm.pdf(x, avg, std)
    kurt = weighted_kurtosis(returns[1:], wts=np.ones_like(returns[1:]))
    skew = weighted_skew(returns[1:], wts=np.ones_like(returns[1:]))
    return_label = "Returns"
    initialize_fig = True
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False
    if "return_label" in kwargs:
        return_label = kwargs["return_label"]

    if "x_lim" in kwargs:
        x_lim = kwargs["x_lim"]
    if "y_lim" in kwargs:
        y_lim = kwargs["y_lim"]

    if initialize_fig:
        fig, ax = plt.subplots()

    x_vals, y_vals = sns.kdeplot(x=returns.values.flatten(), color="cornflowerblue",
                                 weights=np.ones_like(returns.values.flatten()), label="Returns",
                                 alpha=1, ax = ax).get_lines()[0].get_data()
    ax.cla()
    ax.plot(x_vals, y_vals / norm_dist.max(), label=f"{return_label}", color="cornflowerblue", lw=.75)
    ax.plot(x, norm_dist / norm_dist.max(), label="Normal Dist", ls="--", color="red", lw=0.5)
    ax.legend()
    ax.set(xlim=(-0.0075, 0.0075), ylim=(0, y_vals.max() / norm_dist.max() + 0.1), xlabel="Return", ylabel="Density")

    ax.text(loc[0], loc[1], f"kurt = {kurt:.1f}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(loc[0], loc[1] - 0.05, f"skew = {skew:.1f}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


def fan_chart(x: ndarray, y:ndarray, **kwargs):

    """
    From Python for the financial economist at CBS. https://github.com/staxmetrics/python_for_the_financial_economist
    Plots a fan chart.

    If the number of rows of `y` is divisible by 2, the middle row of `y` is plotted as a line in the middle

    Parameters
    ----------
    x: ndarray
        Vector representing the "x-values" of the plot
    y: ndarray
        Matrix of data to plot. Number of columns equal to the length of `x`. Number of rows / 2 is equal to the number
        different colored areas in the plot. It is assumed that values in the first row is smaller than the values in the
        second row and so on.
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

            import numpy as np
            from corelib.plotting import fan_chart
            data = np.array([np.random.normal(size=1000) * s for s in np.arange(0, 1, 0.1)])
            percentiles = np.percentile(data, [10, 20, 50, 80, 90], axis=1)
            fan_chart(np.arange(1, 11, 1), percentiles, labels=['80% CI', '60% CI', 'median'])
            plt.show()

    """

    # defaults
    color_perc = "blue"
    color_median = "red"
    xlabel = None
    ylabel = None
    title = None
    labels = None
    initialize_fig = True

    if 'color' in kwargs:
        color_perc = kwargs['color']
    if 'color_median' in kwargs:
        color_median = kwargs['color_median']
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'labels' in kwargs:
        labels = True
        labels_to_plot = kwargs['labels']
    if "fig" in kwargs:
        fig = kwargs["fig"]
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False

    number_of_rows = y.shape[0]
    number_to_plot = number_of_rows // 2

    if labels is None:
        labels_to_plot = ["" for i in range(number_to_plot + number_of_rows % 2)]

    if initialize_fig:
        fig, ax = plt.subplots()

    for i in range(number_to_plot):

        # for plotting below
        values1 = y[i, :]
        values2 = y[i + 1, :]

        # for plotting above
        values3 = y[-2 - i, :]
        values4 = y[-1 - i, :]

        # calculate alpha
        alpha = 0.95 * (i + 1) / number_to_plot

        ax.fill_between(x, values1, values2, alpha=alpha, color=color_perc, label=labels_to_plot[i])
        ax.fill_between(x, values3, values4, alpha=alpha, color=color_perc)

    # plot center value with specific color
    if number_of_rows % 2 == 1:
        ax.plot(x, y[number_to_plot], color=color_median, label=labels_to_plot[-1])

    # add title
    plt.title(title)
    # add label to x axis
    plt.xlabel(xlabel)
    # add label to y axis
    plt.ylabel(ylabel)
    # legend
    if labels:
        ax.legend()




color_map = plt.cm.get_cmap('tab20c')

default_colors = dict()
default_colors["cornflower"] = "cornflowerblue"
# default_colors['green'] = "green"
#default_colors['light_green'] = '#a8e6cf'
default_colors['red'] = '#ff8b94'
default_colors["medgreen"] = "mediumseagreen"
default_colors["yellow"] = "khaki"
default_colors['black'] = 'black'


default_colors['cyan'] = '#76b4bd'
default_colors['orange'] = '#ffd3b6'


default_colors['light_red'] = '#ffaaa5'
default_colors['gray'] = '#A9A9A9'
default_colors['light_cyan'] = '#bdeaee'

default_colors['dark_blue'] = '#3b7dd8'
default_colors['light_blue'] = '#4a91f2'
default_colors['very_light_blue'] = '#8dbdff'
