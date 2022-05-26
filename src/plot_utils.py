
import matplotlib.patches as patches
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
from src.utils import print_fl


def plot_density(data, ax=None, color='red', arange=None, 
    alpha=1., zorder=1, fill=False, bw=10, neg=False, 
    mult=1.0, y_offset=0, flip=False, lw=1, linestyle='solid', label=None):

    from sklearn.neighbors import KernelDensity
    def _kde_sklearn(x, x_grid, bandwidth):
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        pdf = np.exp(log_pdf)
        return pdf

    if ax is None:
        ax = plt.gca()

    if arange is None:
        arange = min(data), max(data), 1

    x = np.arange(arange[0], arange[1], arange[2])

    y = _kde_sklearn(data, x, bw) * mult
    d = scipy.zeros(len(y))
    fill_mask = y >= d

    if fill:
        if not flip:
            ax.fill_between(x, y+y_offset, 0, color=color,
                     alpha=alpha, linewidth=1, zorder=zorder)
        else:
            ax.fill_betweenx(x, y+y_offset, 0, color=color,
                     alpha=alpha, linewidth=1, zorder=zorder)
    else:
        if not flip:
            ax.plot(x, y+y_offset, color=color,
                 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
                 solid_joinstyle='round', linestyle=linestyle)
        else:
            ax.plot(y+y_offset, x, color=color,
                 alpha=alpha, linewidth=lw, zorder=zorder, label=label,
                 solid_joinstyle='round', linestyle=linestyle)

    return y

def apply_global_settings(titlepad=10, linewidth=1, dpi=100):

    from matplotlib import rcParams
    rcParams['axes.titlepad'] = titlepad 

    # set font globally
    from matplotlib import rcParams
    rcParams['figure.dpi'] = dpi
    rcParams['font.family'] = 'Open Sans'
    rcParams['font.weight'] = 'regular'
    rcParams['figure.titleweight'] = 'regular'
    rcParams['axes.titleweight'] = 'regular'
    rcParams['axes.labelweight'] = 'regular'
    rcParams['axes.labelsize'] = 13
    rcParams['axes.linewidth'] = linewidth
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11    

    rcParams['ytick.major.width'] = linewidth * 3./4
    rcParams['xtick.major.width'] = linewidth * 3./4
    rcParams['ytick.major.size'] = linewidth*2.5
    rcParams['xtick.major.size'] = linewidth*2.5

    rcParams['ytick.minor.width'] = linewidth * 3./4
    rcParams['xtick.minor.width'] = linewidth * 3./4
    rcParams['ytick.minor.size'] = linewidth*1
    rcParams['xtick.minor.size'] = linewidth*1

    rcParams['axes.labelpad'] = 6


def hide_spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_density_scatter(x, y,  bw=(5, 15), cmap='magma_r', vmin=None, 
    vmax=None, ax=None, s=2, alpha=1., zorder=1):
    """
    Plot a scatter plot colored by the smoothed density of nearby points.
    """

    # Perform kernel density smoothing to compute a density value for each
    # point
    try:
        kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
        z = kde.pdf([x, y])
    except ValueError:
        z = np.array([0] * len(x))

    # Use the default ax if none is provided
    if ax is None: ax = plt.gca()

    sorted_idx = np.argsort(z)

    # Plot the outer border of the points in gray
    data_scatter = ax.scatter(x, y, color='none', edgecolor='#c0c0c0',
        s=2, zorder=zorder, cmap=cmap, rasterized=True)
    
    # Reindex the points by the sorting
    z = z[sorted_idx]
    x = x[sorted_idx]
    y = y[sorted_idx]

    # plot the points
    ax.scatter(x, y, c=z, s=3, zorder=zorder, edgecolors='none', 
        cmap=cmap, rasterized=True)

    return data_scatter


def plot_rect(ax, x, y, width, height, color=None, facecolor=None, 
    edgecolor=None, ls='solid', fill_alpha=1., zorder=40, lw=0.0, 
    inset=(0.0, 0.0), fill=True, joinstyle='round'):
    """
    Plot a rectangle for ORF plotting
    """

    if edgecolor is None: edgecolor = color
    if facecolor is None: facecolor = color

    patch = ax.add_patch(
                    patches.Rectangle(
                        (x, y + inset[1]/2.0),   # (x,y)
                        width, height - inset[1], # size
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        lw=lw,
                        joinstyle=joinstyle,
                        ls=ls,
                        fill=fill,
                        alpha=fill_alpha,
                        zorder=zorder
                    ))
