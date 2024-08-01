import numpy as np
from scipy.stats import pearsonr
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling

results = {}

a1 = 2.8e-4
a2 = 2.8e-4
size = 100  # size of the 2D grid
dx = 2. / size  # space step
dims = np.arange(1,9)
lib_sizes = np.arange(10,101,30)
lib_size = 100

sample = 100
count = 0
s = 0


while count <= sample:
    np.random.seed(seed=s)
    X_rand = np.random.rand(size, size)
    Y_rand = np.random.rand(size, size)
    X, Y = run_sim(X_rand, Y_rand, T=30, c=0.1, a1=a1, a2=a2, plot=False)
    correlation_coefficient, p_value = pearsonr(X.flatten(), Y.flatten())
    s += 1
    if 0.2 < correlation_coefficient < 0.35:
        emb = run_optEmbedding_sampling(X, Y, lib_size, dims, cores=6)
        conv = run_GCCM_sampling(X, Y, lib_sizes, E=5, cores=6)
        results_dict[s] = (emb, conv)
        count += 1