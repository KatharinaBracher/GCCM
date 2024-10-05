import numpy as np
from scipy.stats import pearsonr
from diffusion import remove_linear_signals, run_sim
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling

def run_versions_(a1, a2, c, sample, T=30):
    results = {}
    
    size = 100  # size of the 2D grid
    dx = 2. / size  # space step
    dims = np.arange(1,9)
    lib_sizes = np.arange(10,101,30)
    lib_size = 100
    
    sample = sample
    #count = 0
    #s = 0

    print(a1,a2)
    
    #while count <= sample:
    for s in range(sample):
        np.random.seed(seed=s)
        X_rand = np.random.rand(size, size)
        Y_rand = np.random.rand(size, size)
        X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
        correlation_coefficient, p_value = pearsonr(X.flatten(), Y.flatten())
        
        #if correlation_coefficient < 0.35:
        print('running with seed', s)
        emb = run_optEmbedding_sampling(X, Y, lib_size, dims, cores=6)
        conv = run_GCCM_sampling(X, Y, lib_sizes, E=5, cores=6)
        results[s] = {'corr':[correlation_coefficient, p_value], 
                           'optE': emb, 
                           'convergence':conv}
        print('count ', s)
        #count += 1
        #s += 1

    return results