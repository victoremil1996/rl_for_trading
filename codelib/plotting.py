import numpy as np
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

    initialize_fig = True
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False

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
    ax.plot(x_vals, y_vals / y_vals.max(), label="Returns", color="cornflowerblue", lw=.75)
    ax.plot(x, norm_dist / y_vals.max(), label="Normal Dist", ls="--", color="red", lw=0.5)
    ax.legend()
    ax.set(xlim=(-0.0075, 0.0075), ylim=(0, 1), xlabel="Return", ylabel="Density")

    ax.text(loc[0], loc[1], f"kurt = {kurt:.1f}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(loc[0], loc[1] - 0.05, f"skew = {skew:.1f}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


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
