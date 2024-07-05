import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from multiprocessing import Pool

import basic_gao as basic


def run_GCCM_sampling(xMatrix, yMatrix, lib_sizes, E, cores=None):

    print('x_xmap_y')
    x_xmap_y_all, x_xmap_y_results = GCCM(xMatrix, yMatrix, lib_sizes, E, cores=cores)
    print('y_xmap_x')
    y_xmap_x_all, y_xmap_x_results = GCCM(yMatrix, xMatrix, lib_sizes, E, cores=cores)

    results = {'x_xmap_y': x_xmap_y_results, 'y_xmap_x': y_xmap_x_results}    
    #x_xmap_y_all.to_csv(outfile+'x_xmap_y.csv', index=False)  
    #y_xmap_x_all.to_csv(outfile+'y_xmap_x.csv', index=False)  
    
    #with open(outfile, 'wb') as pickle_file:
    #    pickle.dump(results, pickle_file)
    return results


def GCCM(sourceMatrix, targetMatrix, lib_sizes, E, cores=None):
    totalRow, totalCol = sourceMatrix.shape
    target = targetMatrix.flatten()

    # select prediction indices
    # To save the computation time, not every pixel is predict. 
    # The results are almost the same due to the spatial autodim(correctional 
    pred_idx_ = basic.indices_array(totalRow,totalCol)[4::5 , 4::5, ].reshape(-1,2) # filter every 5th pixel tp predict
    pred_idx = np.array([np.ravel_multi_index(pred_idx_[i], (totalRow,totalCol)) for i in range(pred_idx_.shape[0])])
    # ensure prediction indices do not correspond to NAN values
    if np.isnan(target).any():
        nan_idx = np.where(np.isnan(target))[0]
        pred_idx = pred_idx[~np.isin(pred_idx, nan_idx)]
    
    # construct embedding
    print('Constructing embedding')
    embedding = get_embedding(sourceMatrix, E) 

    # initialize 
    xmap_all = pd.DataFrame()
    xmap_results = {}

    if cores is None:
        for lib_size in lib_sizes:
            xmap = GCCMSingle(embedding, sourceMatrix, target, lib_size, totalCol, E)
            xmap_all = pd.concat([xmap_all, xmap])
            xmap_results[lib_size] = basic.results(xmap, pred_idx)

    else:
        with Pool(cores) as p:
            inputs_x = [[embedding, sourceMatrix, target, lib_size, pred_idx, E] for lib_size in lib_sizes]
            xmap = p.starmap(get_xmap, inputs_x)
            
            for (out, lib_size) in xmap:
                xmap_all = pd.concat([xmap_all, out])
                xmap_results[lib_size] = basic.results(out, pred_idx)
        
    return xmap_all, xmap_results

def get_xmap(embedding, sourceMatrix, target, lib_size, pred_idx, E):
    xmap = GCCMSingle(embedding, sourceMatrix, target, lib_size, pred_idx, E)
    
    return xmap, lib_size


def GCCMSingle(embedding, sourceMatrix, target, lib_size, pred_idx, E):
    K = 5 # number of subsamples per iteration
    xmap = pd.DataFrame()

    # sliding library window
    for i in range(K):
        source_indices = np.arange(sourceMatrix.size)
        lib_idx = np.random.choice(source_indices, size=lib_size**2, replace=True)

        # Skips to the next iteration if more than half of the values in the library indices are NA
        if sum(np.isnan(target[lib_idx])) <= (lib_size * lib_size) / 2:
            pred, stats = projection(embedding, target, lib_idx, lib_size, pred_idx, E)
            xmap = pd.concat([xmap, pd.DataFrame([{'L': lib_size, 'rho': stats['rho']}])], ignore_index=True)
        else:
            print('skipped', r, c)

    return xmap


################################
#          Embedding           #
################################

def expand_matrix(dataMatrix, lagNum):
    # Pad matrix with NA values (numpy.nan)
    return np.pad(dataMatrix, pad_width=lagNum, mode='constant', constant_values=np.nan)

def get_lag_indices(window_dim):
    # used to assign indices to variables of each lag order
    if window_dim % 2 == 0:
        raise ValueError("window_dim must be odd")

    # Create a coordinate grid
    center = window_dim // 2
    y, x = np.ogrid[:window_dim, :window_dim]

    # Calculate the Manhattan distance from the center
    dist = np.maximum(np.abs(y - center), np.abs(x - center))

    return dist.flatten()

def get_lagged_variables(dataMatrix, E):
    # sliding window to get lagged variables for each focal unit
    totalRow, totalCol = dataMatrix.shape

    # extracting window of size maxlag around focul unit
    window_dim = E*2 + 1 # window dimension around focal unit
    window = (window_dim,window_dim) # window around focal unit
    
    # pad matrix with nans
    dataMatrix = expand_matrix(dataMatrix, E) 

    # arrays of neighbors for each unit
    laggedVar = sliding_window_view(dataMatrix, window) # gets window around each unit
    laggedVar = laggedVar.reshape(totalRow*totalCol,window_dim*window_dim) # reshape to flatten array

    # layered window indicating position of lag orders (0 focal unit,1, 2,...)
    lag_indices = get_lag_indices(window_dim)
    return laggedVar, lag_indices

def get_embedding(dataMatrix, E):
    # get neighbors of each unit and position indices of lag orders
    laggedVar, lag_indices = get_lagged_variables(dataMatrix, E) 
    
    # add focal units s
    embedding = [dataMatrix.flatten()]  # flatten the original dataMatrix (row-major)
    
    # higher order neighbors
    for i in range(1, E):
        lag = laggedVar[:,np.where(lag_indices==i)] # extract neighbors of different order
        embedding.append(lag.squeeze())
        
    return embedding



################################
#          Projection          #
################################


def projection(embedding, target, lib_idx, lib_size, pred_idx, E):
    # account for different image size and unexpected behavior in R

    pred = np.full_like(target, np.nan)
    #pred = np.full_like(target, np.nan).flatten()

    for p in pred_idx:
        # removes the prediction point from the library to ensure the model does not use its own value in prediction
        lib_idx_ = lib_idx.copy()
        lib_idx_ = lib_idx_[~np.isin(lib_idx_, p)]

        # compute distances between the embedding of the prediction point and embedding of all points in the adjusted library.
        distances = get_distances(embedding, lib_size, E, p, lib_idx_)
        
        # find nearest neighbors
        neighbors = np.argsort(distances, kind='stable')[:E+1]
        min_distance = distances[neighbors[0]]
        
        if np.isnan(min_distance):
            continue
        elif min_distance == 0: # perfect match
            weights = np.full((E+1), 0.000001)
            weights = np.where(distances[neighbors] == 0, 1, weights)
        else:
            weights = np.exp(-distances[neighbors] / min_distance)
            weights = np.where(weights <0.000001, 0.000001, weights)
            
        total_weight = np.sum(weights)
        
        # make prediction
        pred[p] = np.dot(weights, target[lib_idx_[neighbors]]) / total_weight 

    #print(basic.compute_stats(target[row, col], pred[np.where(pred_idx)[0]]))
    return pred, basic.compute_stats(target[pred_idx], pred[pred_idx]) 


def get_distances(embedding, lib_size, E, p, lib_idx):
    distances = np.full((lib_idx.shape[0], E), np.inf) 
    emb = embedding[0]
    distances[:,0] = abs(emb[lib_idx] - emb[p])
    
    for e in range(1, len(embedding)):
        emb = embedding[e]
        dist = emb[lib_idx,] - emb[p,]
        distances[:,e] = abs(np.nanmean(dist, axis=1)) # take mean over neighbors
    return np.nanmean(distances, axis=1) # take mean over dimensions
