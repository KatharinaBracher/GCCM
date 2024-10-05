import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from multiprocessing import Pool

import basic_gao as basic

import GCCM_sampling as GCCM

def run_optEmbedding_sampling(xMatrix, yMatrix, lib_size, dims, cores=None):
    print('x_xmap_y')
    x_xmap_y_all, x_xmap_y_results = GCCM_optEmbedding(xMatrix, yMatrix, lib_size, dims, cores=cores)
    print('y_xmap_x')
    y_xmap_x_all, y_xmap_x_results = GCCM_optEmbedding(yMatrix, xMatrix, lib_size, dims, cores=cores)

    results = {'x_xmap_y': x_xmap_y_results, 'y_xmap_x': y_xmap_x_results}    
    #x_xmap_y_all.to_csv(outfile+'_x_xmap_y_optE.csv', index=False)  
    #y_xmap_x_all.to_csv(outfile+'_y_xmap_x_optE.csv', index=False)  
    
    #with open(outfile+'.pkl', 'wb') as pickle_file:
    #    pickle.dump(results, pickle_file)
    return results


def GCCM_optEmbedding(sourceMatrix, targetMatrix, lib_size, dims, cores=None):
    totalRow, totalCol = sourceMatrix.shape
    target = targetMatrix.flatten()

    # FILTER
    # To save the computation time, not every pixel is predict. 
    # The results are almost the same due to the spatial autodim(correctional 
    pred_idx_ = basic.indices_array(totalRow,totalCol)[4::5 , 4::5, ].reshape(-1,2) # filter every 5th pixel tp predict
    pred_idx = np.array([np.ravel_multi_index(pred_idx_[i], (totalRow,totalCol)) for i in range(pred_idx_.shape[0])])
    # ensure prediction indices do not correspond to NAN values
    if np.isnan(target).any():
        nan_idx = np.where(np.isnan(target))[0]
        pred_idx = pred_idx[~np.isin(pred_idx, nan_idx)]

    # initialize 
    xmap_all = pd.DataFrame()
    xmap_results = {}
    
    # construct embedding
    Embeddings = get_allEmbedding(sourceMatrix, dims)        

    if cores is None:
        for E, embedding in zip(dims, Embeddings):
            #print(E)
            xmap = GCCM.GCCMSingle(embedding, sourceMatrix, target, lib_size, pred_idx, E)
            xmap_all = pd.concat([xmap_all, xmap])
            xmap_results[E] = basic.results(xmap, pred_idx)

    else:
        with Pool(cores) as p:
            inputs_x = [[embedding, sourceMatrix, target, lib_size, pred_idx, E] for E, embedding in zip(dims, Embeddings)]
            xmap = p.starmap(get_xmap, inputs_x)
            
            for (out, E) in xmap:
                xmap_all = pd.concat([xmap_all, out])
                xmap_results[E] = basic.results(out, pred_idx)
        
    return xmap_all, xmap_results


def get_xmap(embedding, sourceMatrix, target, lib_size, pred_idx, E):
    #print('dimension ', E)
    xmap = GCCM.GCCMSingle(embedding, sourceMatrix, target, lib_size, pred_idx, E)
    
    return xmap, E

def get_allEmbedding(sourceMatrix, dims):
    print('Constructing embedding')
    return [GCCM.get_embedding(sourceMatrix, E) for E in dims]

