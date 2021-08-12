"""
Performance benchmarking for parallel crand

...

python crand_perf.py BRANCH DRAWS PERMUTATIONS CORES
"""

import os, sys, time, datetime
import subprocess
import geopandas, pandas
import numpy as np
import crand
from time import time
from esda.moran import _moran_local_crand
from libpysal import examples, weights

# Print versions
import numba, joblib
print((
    f"{datetime.datetime.now()} | "\
    f"Numba: {numba.__version__} | "\
    f"Joblib: {joblib.__version__}"
       ))

# Parse arguments
BRANCH = sys.argv[1]
DRAWS = int(sys.argv[2])
PERMUTATIONS = int(sys.argv[3])
CORES = int(sys.argv[4])
if int(CORES) == -1:
    CORES = os.cpu_count()
SEED = 12345

# Checkout branch
#subprocess.run(["git", "checkout", BRANCH])
#print(f"Branch {BRANCH} loaded")

# Load data
_ = examples.load_example("NCOVR")
var = "HR60"
db = geopandas.read_file(
        examples.get_path("NAT.shp")
)
## Augment size
db = pandas.concat([db]*10)
w = weights.Queen.from_dataframe(db)
w.transform = "R"

z = db[var].values
z = (z - z.mean()) / z.std()

zl = weights.lag_spatial(w, z)
observed = (w.n - 1) * z * zl / (z * z).sum()

cardinalities = np.array((w.sparse != 0).sum(1)).flatten()

weights = w.sparse.data

permuted_ids = crand.vec_permutations(
        cardinalities.max(), w.n, PERMUTATIONS, SEED
)

scaling = (w.n - 1) / (z * z).sum()

n_jobs = CORES

keep = False

stat_func = _moran_local_crand

# Loop over executions (DRAWS)
compiler = crand.parallel_crand(
        z, 
        observed, 
        cardinalities,
        weights,
        permuted_ids,
        scaling,
        n_jobs,
        keep,
        stat_func,
)
print((
    f"Benchmarking {PERMUTATIONS} permutations using "\
    f"{CORES} cores and {DRAWS} reps..."
))
ts = []
for i in range(DRAWS):
    t0 = time()
    compiler = crand.parallel_crand(
            z, 
            observed, 
            cardinalities,
            weights,
            permuted_ids,
            scaling,
            n_jobs,
            keep,
            stat_func,
    )
    t1 = time()
    t = t1 - t0
    ts.append(t)
    #print(f"\tRep {i+1}: {np.round(t, 4)} seconds")
ts = np.array(ts)
print((
    f"\n{PERMUTATIONS} perms | {CORES} cores | "\
    f"N: {w.n} | "\
    f"Mean {np.round(ts.mean(), 4)}s | "\
    f"Std: {np.round(ts.std(), 4)}s\n"
))

