import numpy as np
from scipy.stats import pearsonr
from scipy.stats import t


def indices_array(m,n):
    r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

def convert_column_major_idx(emb, i):
    num_rows = emb.shape[0]
    # Calculate the row and column in column-major order
    row = i % num_rows
    col = i // num_rows

    return (row, col)

def compute_stats(obs, pred):
    """
    Computes performance metrics for how well predictions match observations.
    
    Parameters:
    obs (array-like): Vector of observations
    pred (array-like): Vector of predictions
    
    Returns:
    DataFrame: A data frame containing N, rho, mae, and rmse
    """
    # Ensure obs and pred are numpy arrays
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    
    # Mask for finite values
    mask = np.isfinite(obs) & np.isfinite(pred)
    
    # Calculate N
    N = np.sum(mask)
    
    # Calculate rho (correlation coefficient)
    if N > 1:
        rho = np.corrcoef(obs[mask], pred[mask])[0, 1]
    else:
        rho = np.nan
    
    # Calculate mae (mean absolute error)
    mae = np.mean(np.abs(obs[mask] - pred[mask]))
    
    # Calculate rmse (root mean square error)
    rmse = np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2))
    
    # Create a DataFrame to return the results
    result = {
        'N': N,
        'rho': rho,
        'mae': mae,
        'rmse': rmse}

    return result


def significance(r, n):
    # Calculate the t-value
    t_value = r * np.sqrt((n - 2) / (1 - r**2))
    
    # Calculate the two-tailed p-value
    p_value = (1-t.cdf(np.abs(t_value), df=n-2))
    
    return p_value #*2 !!!!!!!!!

from scipy.stats import norm

def confidence(r, n, level=0.05):
    # Fisher transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    ztheta = 1 / np.sqrt(n - 3)
    
    # Critical value for the normal distribution
    qZ = norm.ppf(1 - level / 2)
    
    # Calculate upper and lower bounds in z space
    upper = z + qZ * ztheta
    lower = z - qZ * ztheta
    
    # Transform bounds back to r space
    r_upper = (np.exp(2 * upper) - 1) / (np.exp(2 * upper) + 1)
    r_lower = (np.exp(2 * lower) - 1) / (np.exp(2 * lower) + 1)
    
    return r_upper, r_lower

def results(df, pred):
    n = pred.shape[0]
    mean = df["rho"].mean()
    sig = significance(mean, n)
    conf = confidence(mean, n, level=0.05)
    return {'mean': mean, 'sig': sig, 'conf': conf}
    








