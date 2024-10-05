import numpy as np
from scipy.stats import pearsonr
from diffusion import remove_linear_signals, run_sim
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling
import pickle
from multiprocessing import Pool

def run_sample(sample, size, c, a1, a2):
    results = {}
    filename = 'results_test/c'+str(c)+'_a1'+str(a1)+'_a2'+str(a2)+'.pkl'

    print('running with c=', c, 'and a=', a1, a2)
    
    for s in range(sample):
        np.random.seed(seed=s)
        #X_rand = np.random.rand(size, size)
        #Y_rand = np.random.rand(size, size)
        #X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
        #correlation_coefficient, p_value = pearsonr(X.flatten(), Y.flatten())
        #if correlation_coefficient < 0.35:
        print('running with seed', s)
        #conv = run_GCCM_sampling(X, Y, lib_sizes, E=5, cores=6)
        conv = None
        correlation_coefficient, p_value = None, None
        results[s] = {'corr':[correlation_coefficient, p_value], 
                      'gccm':conv}
        
    with open(filename, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
            
def run_grid():

    T = 30
    sample = 50
    size = 100  # size of the 2D grid
    dx = 2. / size  # space step
    dims = np.arange(1,9)
    lib_sizes = [100]

    c_list = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
     0.11, 0.12, 0.13, 0.14]
    a1 = np.linspace(2.8e-3, 2.8e-5,15)
    a2 = np.flip(a1)
    a_list = np.dstack((a1,a2)).squeeze() 
    
    with Pool() as p:
        p.starmap(run_sample, [(sample, size, c ,a1 ,a2) for c in c_list for (a1, a2) in a_list])
