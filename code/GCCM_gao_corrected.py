import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from multiprocessing import Pool

import basic_gao as basic


def run_GCCM_corrected(xMatrix, yMatrix, lib_sizes, E, cores=None):
    totalRow, totalCol = xMatrix.shape
    # To save the computation time, not every pixel is predict. 
    # The results are almost the same due to the spatial autodim(correctional 
    pred = basic.indices_array(totalRow,totalCol)[4::5 , 4::5, ].reshape(-1,2) # filter every 5th pixel
    print('x_xmap_y')
    x_xmap_y_all, x_xmap_y_results = GCCM(xMatrix, yMatrix, pred, lib_sizes, E, cores=cores)
    print('y_xmap_x')
    y_xmap_x_all, y_xmap_x_results = GCCM(yMatrix, xMatrix, pred, lib_sizes, E, cores=cores)

    results = {'x_xmap_y': x_xmap_y_results, 'y_xmap_x': y_xmap_x_results}    
    #x_xmap_y_all.to_csv(outfile+'x_xmap_y.csv', index=False)  
    #y_xmap_x_all.to_csv(outfile+'y_xmap_x.csv', index=False)  
    
    #with open(outfile, 'wb') as pickle_file:
    #    pickle.dump(results, pickle_file)
    return results


def GCCM(sourceMatrix, targetMatrix, pred, lib_sizes, E, cores=None):
    totalRow, totalCol = sourceMatrix.shape
    target = targetMatrix.flatten() ################################## 
    
    # construct embedding
    print('Constructing embedding')
    embedding = get_embedding(sourceMatrix, E) 

    # initialize 
    xmap_all = pd.DataFrame()
    xmap_results = {}

    if cores is None:
        for lib_size in lib_sizes:
            xmap = GCCMSingle(embedding, target, lib_size, pred, totalRow, totalCol, E)
            xmap_all = pd.concat([xmap_all, xmap])
            xmap_results[lib_size] = basic.results(xmap, pred)

    else:
        with Pool(cores) as p:
            inputs_x = [[embedding, target, lib_size, pred, totalRow, totalCol, E] for lib_size in lib_sizes]
            xmap = p.starmap(get_xmap, inputs_x)
            
            for (out, lib_size) in xmap:
                xmap_all = pd.concat([xmap_all, out])
                xmap_results[lib_size] = basic.results(out, pred)
        
    return xmap_all, xmap_results

def get_xmap(embedding, target, lib_size, pred, totalRow, totalCol, E):
    #sprint('libsize ', lib_size)
    xmap = GCCMSingle(embedding, target, lib_size, pred, totalRow, totalCol, E)
    
    return xmap, lib_size


def GCCMSingle(embedding, target, lib_size, pred, totalRow, totalCol, E):
    xmap = pd.DataFrame()
    pred_flat = [np.ravel_multi_index(pred[i], (totalRow,totalCol)) for i in range(pred.shape[0])]

    # sliding library window
    for r, c in product(range(totalRow - lib_size + 1), range(totalCol - lib_size + 1)):
        # initialize flatten mask arrays
        pred_indices = np.zeros(totalRow * totalCol, dtype=bool)
        lib_indices = np.zeros(totalRow * totalCol, dtype=bool)

        pred_indices[pred_flat] = True # mask which pixels in the total matrix should be predicted
        if np.isnan(target).any():
            pred_indices[np.isnan(target)] = False # ensure prediction indices do not correspond to NAN values in yPred

        # mask for pixels in current library
        lib_rows = np.arange(r, r + lib_size)
        lib_cols = np.arange(c, c + lib_size)
        lib_ids = np.array(list(product(lib_rows, lib_cols)))
        lib_ids_flat = [np.ravel_multi_index(lib_ids[i], (totalRow,totalCol)) for i in range(lib_ids.shape[0])]
        lib_indices[lib_ids_flat] = True

        # Skips to the next iteration if more than half of the values in the library indices are NA
        if sum(np.isnan(target[np.where(lib_indices)])) <= (lib_size * lib_size) / 2:
            pred, stats = projection(embedding, target, lib_indices, lib_size, pred_indices, E)
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
    ############################################################################# embedding = [dataMatrix.T]
    embedding = [dataMatrix.flatten()]  # flatten the original dataMatrix (row-major)

    # s(1), s(2), ..., s(E-1)
    for i in range(1, E): ############################################################################# E+1
        lag = laggedVar[:,np.where(lag_indices==i)] # extract neighbors of different order
        embedding.append(lag.squeeze())
        # [0] row*col
        # [1] row*col x 8 1st order
        # [2] row*col x 16 2nd order, ...
    return embedding



################################
#          Projection          #
################################


def projection(embedding, target, lib_indices, lib_size, pred_indices, E):
    # account for different image size and unexpected behavior in R

    pred = np.full_like(target, np.nan)
    #pred = np.full_like(target, np.nan).flatten()

    for p in np.where(pred_indices)[0]:
        # removes the prediction point from the library to ensure the model does not use its own value in prediction
        lib_indices_ = lib_indices.copy()
        lib_indices_[p] = False

        libs = np.where(lib_indices_)[0] # get indices

        # compute distances between the embedding of the prediction point and embedding of all points in the adjusted library.

        distances = get_distances(embedding, lib_size, E, p, libs)
        
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
        # weighted average of the target values at the neighbor locations, using the calculated weights
        ################################################# change from R col major flattening to row major indexing
        pred[p] = np.dot(weights, target[libs[neighbors]]) / total_weight #######################

    #print(basic.compute_stats(target[row, col], pred[np.where(pred_indices)[0]]))
    return pred, basic.compute_stats(target[np.where(pred_indices)[0]], pred[np.where(pred_indices)[0]]) ####################


def get_distances(embedding, lib_size, E, p, libs):
    distances = np.full((libs.shape[0], E), np.inf) 
    emb = embedding[0]
    distances[:,0] = abs(emb[libs] - emb[p])
    
    for e in range(1, len(embedding)):
        emb = embedding[e]
        dist = emb[libs,] - emb[p,]
        distances[:,e] = abs(np.nanmean(dist, axis=1)) # take mean over neighbors
    return np.nanmean(distances, axis=1) # take mean over dimensions
#############################################################################
    '''
    distances = np.full((libs.shape[0], E+1), np.inf)
    for e in range(len(embedding)):
        emb = embedding[e]
    
        num_rows = emb.shape[0]
        # Calculate the row and column in column-major order
        row, col = basic.convert_column_major_idx(emb, p)
        row_lib, col_lib = basic.convert_column_major_idx(emb, libs)
        distances[:,e] = abs(emb[row_lib, col_lib] - emb[row, col])
    '''