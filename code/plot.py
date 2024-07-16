import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import pearsonr
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Set specific font sizes
plt.rcParams.update({
    'font.size': 12,          # Global font size
    'axes.titlesize': 11,     # Title font size
    'axes.labelsize': 11,     # X and Y axis labels font size
    'xtick.labelsize': 9,    # X-axis tick labels font size
    'ytick.labelsize': 9,    # Y-axis tick labels font size
    'legend.fontsize': 10,    # Legend font size
})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

dims = np.arange(1,9)
lib_sizes = np.arange(10,101,30)
lib_size = 100

def get_mean(results, arr):
    xy = []
    yx = []
    for i in arr:
        xy.append(results['x_xmap_y'][i]['mean'])
        yx.append(results['y_xmap_x'][i]['mean'])
    return np.array(xy), np.array(yx)

def get_interval(results, arr):
    xy_u = []
    xy_l = []
    yx_u = []
    yx_l = []
    for i in arr:
        u, l = results['x_xmap_y'][i]['conf']
        xy_u.append(u)
        xy_l.append(l)
        u, l = results['y_xmap_x'][i]['conf']
        yx_u.append(u)
        yx_l.append(l)
    return np.array(xy_u), np.array(xy_l), np.array(yx_u), np.array(yx_l)

# Appendix figure
def make_Eplot(fig, position, results, title=r' ', share=False):
    x_xmap_y, y_xmap_x = get_mean(results, dims)
    xy_u, xy_l, yx_u, yx_l = get_interval(results, dims)

    ax = fig.add_subplot(position)
    ax.plot(dims, y_xmap_x,  c='#FB8500', lw=2, label = r'X $\rightarrow$ Y')
    ax.fill_between(dims, yx_l, yx_u, color='#FB8500', alpha=0.1, lw=0)
    ax.plot(dims, x_xmap_y,  c='#017D84', lw=2, label = r'Y $\rightarrow$ X' )
    ax.fill_between(dims, xy_l, xy_u, color='#017D84', alpha=0.1, lw=0)
    
    ax.set_ylim(-0.1,1)
    ax.set_title(title)
    ax.set_xlabel('E')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    if not share:
        ax.set_ylabel(r'$\rho$')

        legend = ax.legend(frameon=False, handlelength=0.5)
        for legobj in legend.legendHandles:
            legobj.set_linewidth(2)
    if share:
        ax.set_yticklabels([])
    
    return ax

def make_Lplot(fig, position, results, title=None, share=False):
    x_xmap_y, y_xmap_x = get_mean(results, lib_sizes)
    xy_u, xy_l, yx_u, yx_l = get_interval(results, lib_sizes)

    ax = fig.add_subplot(position)
    ax.plot(lib_sizes, y_xmap_x,  c='#D00000', lw=2, label = r'X $\rightarrow$ Y')
    ax.fill_between(lib_sizes, yx_l, yx_u, color='#D00000', alpha=0.1, lw=0)
    ax.plot(lib_sizes, x_xmap_y,  c='#006EBC', lw=2, label = r'Y $\rightarrow$ X' )
    ax.fill_between(lib_sizes, xy_l, xy_u, color='#006EBC', alpha=0.1, lw=0)
    
    ax.set_ylim(-0.1,1)
    if title:
        ax.set_title(title)
    ax.set_xlabel('L')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    if not share:
        ax.set_ylabel(r'$\rho$')

        legend = ax.legend(frameon=False, handlelength=0.5)
        for legobj in legend.legendHandles:
            legobj.set_linewidth(2)
    if share:
        ax.set_yticklabels([])
        
    return ax