import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from plot import get_mean, get_interval, make_Eplot, make_Lplot

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

size = 100  # size of the 2D grid
dx = 2. / size  # space step

def remove_linear_signals(x, y):
    # Flatten the arrays
    x_flat = x.flatten().reshape(-1, 1)
    y_flat = y.flatten().reshape(-1, 1)

    # Fit linear model y = M*x + c
    model = LinearRegression()

    model.fit(y_flat, x_flat)
    x_pred = model.predict(y_flat)
    x_star_flat = x_flat - x_pred
    x_star = x_star_flat.reshape(x.shape)
    
    model.fit(x_flat, y_flat)
    y_pred = model.predict(x_flat)
    y_star_flat = y_flat - y_pred
    y_star = y_star_flat.reshape(y.shape) 

    return x_star, y_star

def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.viridis, vmin=0, vmax=1,
              interpolation='None')
           # extent=[-1, 1, -1, 1])
    ax.set_axis_off()

def run_sim(X_in, Y_in, T, c, a1, a2, saveas=False):
    X = X_in.copy()
    Y = Y_in.copy()
    
    dt = .001  # time step, 
    n = int(T / dt)  # number of iterations

    fig, axes = plt.subplots(2,8, figsize=(6.3, 2))
    step_plot = n // 8
    # We simulate the PDE with the finite difference
    # method.
    for i in range(n):
        # We compute the Laplacian of u and v.
        deltaX = laplacian(X)
        deltaY = laplacian(Y)
        # We take the values of u and v inside the grid.
        Xc = X[1:-1, 1:-1]
        Yc = Y[1:-1, 1:-1]
        
        # We update the variables.
        X[1:-1, 1:-1], Y[1:-1, 1:-1] = \
            Xc + dt * (a1 * deltaX - Xc**2),\
            Yc + dt * (a2 * deltaY - Yc**2 + c * Xc * Yc)
        
        # Neumann conditions: derivatives at the edges
        # are null.
        for B in (X, Y):
            B[0, :] = B[1, :]
            B[-1, :] = B[-2, :]
            B[:, 0] = B[:, 1]
            B[:, -1] = B[:, -2]
    
        # We plot the state of the system at
        # 9 different times.
        if i % step_plot == 0 and i < 8 * step_plot:
            ax1 = axes[0, i // step_plot]
            ax2 = axes[1, i // step_plot]
            
            show_patterns(X, ax=ax1)
            ax1.set_title(f'${i * dt:.0f}$')
            show_patterns(Y, ax=ax2)
            #ax.set_title(f'Y $t={i * dt:.0f}$')
            if i // step_plot == 0:
                ax1.text(-0.25, 0.5, 'X', transform=ax1.transAxes, fontsize=11, horizontalalignment='center')
                ax2.text(-0.25, 0.5, 'Y', transform=ax2.transAxes, fontsize=11, horizontalalignment='center')
                
    plt.tight_layout()
    if saveas:
        plt.savefig(saveas, bbox_inches='tight')
    return X, Y