import os, sys, time, datetime
import subprocess
import geopandas, pandas
import numpy as np
import crand
from time import time
from esda.moran import _moran_local_crand
from libpysal import examples
from libpysal import weights as lpw

SEED = 12345
CPUS = os.cpu_count()
N_FACTORS = [1, 5, 10]
PERMUTATIONS = [99, 999, 9999]
CORES = [1] + list(range(2, CPUS+1, 2))

def run_branch(branch, draws=5, var="HR60"):
    subprocess.run(["git", "checkout", branch])
    print(f"Branch {branch} loaded")
    _ = examples.load_example("NCOVR")
    db = geopandas.read_file(
            examples.get_path("NAT.shp")
    )
    all_times = []
    mean_times = []
    for n_factor in N_FACTORS:
        db = pandas.concat([db]*n_factor)
        w = lpw.Queen.from_dataframe(db)
        w.transform = "R"
        for perms in PERMUTATIONS:
            for n_jobs in CORES:
                # Load data
                z = db[var].values
                z = (z - z.mean()) / z.std()

                zl = lpw.lag_spatial(w, z)
                observed = (w.n - 1) * z * zl / (z * z).sum()

                cardinalities = np.array(
                        (w.sparse != 0).sum(1)
                ).flatten()

                weights = w.sparse.data

                permuted_ids = crand.vec_permutations(
                        cardinalities.max(), w.n, perms, SEED
                )

                scaling = (w.n - 1) / (z * z).sum()

                keep = False

                stat_func = _moran_local_crand
                # Compile burn
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
                ts = []
                for i in range(draws):
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
                    all_times.append([n_factor, perms, n_jobs, t])
                ts = np.array(ts)
                mean_times.append([n_factor, perms, n_jobs, ts.mean()])
                print((
                    f"{perms} perms | {n_jobs} cores | "\
                    f"N: {w.n} | "\
                    f"Mean {np.round(ts.mean(), 4)}s | "\
                    f"Std: {np.round(ts.std(), 4)}s"
                ))
    all_times = pandas.DataFrame(
            all_times, 
            columns=["n_factor", "perms", "n_jobs", "seconds"]
    )
    all_times["branch"] = branch
    mean_times = pandas.DataFrame(
            mean_times, 
            columns=["n_factor", "perms", "n_jobs", "seconds"]
    )
    mean_times["branch"] = branch
    return all_times, mean_times

def sim_over_branches(branches):
    all_times_bag = []
    mean_times_bag = []
    for branch in branches:
        all_times, mean_times = run_branch(branch)
        all_times_bag.append(all_times)
        mean_times_bag.append(mean_times)
    pandas.concat(all_times_bag).to_csv("all_times.csv", index=False)
    pandas.concat(mean_times_bag).to_csv("mean_times.csv", index=False)
    return None

if __name__ == '__main__':
    branches = ['master', 'crand-innerlimit', 'crand-automemmap']
    _ = sim_over_branches(branches)

