import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from multiprocessing import Pool

import basic_gao as basic
import GCCM_gao_corrected as GCCM

def run_optEmbedding(xMatrix, yMatrix, lib_size, dims, outfile, cores=None):
    totalRow, totalCol = xMatrix.shape
    # To save the computation time, not every pixel is predict. 
    # The results are almost the same due to the spatial autodim(correctional 
    pred = basic.indices_array(totalRow,totalCol)[4::5 , 4::5, ].reshape(-1,2) # filter every 5th pixel
    print('x_xmap_y')
    x_xmap_y_all, x_xmap_y_results = GCCM_optEmbedding(xMatrix, yMatrix, pred, lib_size, dims, cores=cores)
    print('y_xmap_x')
    y_xmap_x_all, y_xmap_x_results = GCCM_optEmbedding(yMatrix, xMatrix, pred, lib_size, dims, cores=cores)

    results = {'x_xmap_y': x_xmap_y_results, 'y_xmap_x': y_xmap_x_results}    
    x_xmap_y_all.to_csv(outfile+'_x_xmap_y_optE.csv', index=False)  
    y_xmap_x_all.to_csv(outfile+'_y_xmap_x_optE.csv', index=False)  
    
    with open(outfile+'.pkl', 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
    return results


def GCCM_optEmbedding(sourceMatrix, targetMatrix, pred, lib_size, dims, cores=None):
    totalRow, totalCol = sourceMatrix.shape
    yPred = targetMatrix.flatten()

    # initialize 
    xmap_all = pd.DataFrame()
    xmap_results = {}

    # construct embedding
    Embeddings = get_allEmbedding(sourceMatrix, dims)        

    if cores is None:
        for E, embedding in zip(dims, Embeddings):
            xmap = GCCM.GCCMSingle(embedding, yPred, lib_size, pred, totalRow, totalCol, E)
            xmap_all = pd.concat([xmap_all, xmap])
            xmap_results[E] = basic.results(xmap, pred)

    else:
        with Pool(cores) as p:
            inputs_x = [[embedding, yPred, lib_size, pred, totalRow, totalCol, E] for E, embedding in zip(dims, Embeddings)]
            xmap = p.starmap(get_xmap, inputs_x)
            
            for (out, lib_size) in xmap:
                xmap_all = pd.concat([xmap_all, out])
                xmap_results[E] = basic.results(out, pred)
        
    return xmap_all, xmap_results


def get_xmap(embedding, yxPred, lib_size, pred, totalRow, totalCol, E):
    print('libsize ', lib_size)
    xmap = GCCM.GCCMSingle(embedding, yxPred, lib_size, pred, totalRow, totalCol, E)
    
    return xmap, lib_size

def get_allEmbedding(sourceMatrix, dims):
    print('Constructing embedding')
    return [GCCM.get_embedding(sourceMatrix, E) for E in dims]

