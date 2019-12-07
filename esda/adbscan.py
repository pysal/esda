"""
A-DBSCAN implementation
"""

__author__ = "Dani Arribas-Bel <daniel.arribas.bel@gmail.com>"

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["ADBSCAN", "remap_lbls", "ensemble"]


class ADBSCAN:
    """
    A-DBSCAN, as introduced in Arribas-Bel, Garcia-Lopez & Viladecans-Marsal
    (2020)
    ...

    Parameters
    ----------
    eps         : float
                  The maximum distance between two samples for them to be considered
                  as in the same neighborhood.
    min_samples : int
                  The number of samples (or total weight) in a neighborhood
                  for a point to be considered as a core point. This includes the
                  point itself.
    algorithm   : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
                  The algorithm to be used by the NearestNeighbors module
                  to compute pointwise distances and find nearest neighbors.
                  See NearestNeighbors module documentation for details.
    n_jobs      : int
                  [Optional. Default=1] The number of parallel jobs to run. If
                  -1, then the number of jobs is set to the number of CPU
                  cores.
    pct_exact   : float
                  [Optional. Default=0.1] Percentage of the entire dataset
                  used to calculate DBSCAN in each draw
    reps        : int
                  [Optional. Default=100] Number of random samples to draw in order to
                  build final solution
    keep_solus  : Boolean
                  [Optional. Default=False] If True, the `solus` object is
                  kept, else it is deleted to save memory
    pct_thr     : float
                  [Optional. Default=0.9] Minimum percentage of replications that a non-noise 
                  label need to be assigned to an observation for that observation to be labelled
                  as such
    Attributes
    ----------
    labels_     : array
                  Cluster labels for each point in the dataset given to fit().
                  Noisy (if the proportion of the most common label is < pct_thr) samples are given
                  the label -1.
    votes       : DataFrame
                  Table indexed on `X.index` with `labels_` under the `lbls`
                  column, and the frequency across draws of that label under
                  `pct`
    solus       : DataFrame, shape = [n, reps]
                  Each solution of labels for every draw
    """

    def __init__(
        self,
        eps,
        min_samples,
        algorithm="auto",
        n_jobs=1,
        pct_exact=0.1,
        reps=100,
        keep_solus=False,
        pct_thr=0.9,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.reps = reps
        self.n_jobs = n_jobs
        self.pct_exact = pct_exact
        self.pct_thr = pct_thr
        self.keep_solus = keep_solus

    def fit(self, X, y=None, sample_weight=None, xy=["X", "Y"], multi_cpu=False):
        """
        Perform ADBSCAN clustering from fetaures
        ...

        Parameters
        ----------
        X               : DataFrame
                          Features
        sample_weight   : Series, shape (n_samples,)
                          [Optional. Default=None] Weight of each sample, such
                          that a sample with a weight of at least ``min_samples`` 
                          is by itself a core sample; a sample with negative
                          weight may inhibit its eps-neighbor from being core.
                          Note that weights are absolute, and default to 1.
        xy              : list
                          [Default=`['X', 'Y']`] Ordered pair of names for XY
                          coordinates in `xys`
        y               : Ignored
        multi_cpu       : Boolean
                          [Default=False] If True, paralelise where possible.
                          NOTE: currently not implemented
        """
        n = X.shape[0]
        zfiller = len(str(self.reps))
        solus = pd.DataFrame(
            np.zeros((X.shape[0], self.reps), dtype=str),
            index=X.index,
            columns=["rep-%s" % str(i).zfill(zfiller) for i in range(self.reps)],
        )
        if multi_cpu is True:
            import multiprocessing as mp

            pool = mp.Pool(mp.cpu_count())
            print("On multiple cores...")
            # Set different parallel seeds!!!
            raise NotImplementedError
        else:
            for i in range(self.reps):
                pars = (
                    n,
                    X,
                    sample_weight,
                    xy,
                    self.pct_exact,
                    self.eps,
                    self.min_samples,
                    self.algorithm,
                    self.n_jobs,
                )
                lbls_pred = _one_draw(pars)
                solus.iloc[:, i] = lbls_pred

        self.votes = ensemble(solus, X, xy, multi_cpu=multi_cpu)
        lbls = self.votes["lbls"].values
        lbl_type = type(solus.iloc[0, 0])
        lbls[self.votes["pct"] < self.pct_thr] = lbl_type(-1)
        self.labels_ = lbls
        if not self.keep_solus:
            del solus
        else:
            self.solus = solus
        return self


def _one_draw(pars):
    n, X, sample_weight, xy, pct_exact, eps, min_samples, algorithm, n_jobs = pars
    rids = np.arange(n)
    np.random.shuffle(rids)
    rids = rids[: int(n * pct_exact)]

    X_thin = X.iloc[rids, :]

    thin_sample_weight = None
    if sample_weight is not None:
        thin_sample_weight = sample_weight.iloc[rids]

    dbs = DBSCAN(
        eps=eps,
        min_samples=int(np.round(min_samples * pct_exact)),
        algorithm=algorithm,
        n_jobs=n_jobs,
    ).fit(X_thin[xy], sample_weight=thin_sample_weight)
    lbls_thin = pd.Series(dbs.labels_.astype(str), index=X_thin.index)

    NR = KNeighborsClassifier(n_neighbors=1)
    NR.fit(X_thin[xy], lbls_thin)
    lbls_pred = pd.Series(NR.predict(X[xy]), index=X.index)
    return lbls_pred


def remap_lbls(solus, xys, xy=["X", "Y"], multi_cpu=True):
    """
    Remap labels in solutions so they are comparable (same label
    for same cluster)
    ...

    Arguments
    ---------
    solus       : DataFrame
                  Table with labels for each point (row) and solution (column)
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    multi_cpu   : Boolean
                  [Default=False] If True, paralelise remapping
    
    Returns
    -------
    onel_solus  : DataFrame
    """
    lbl_type = type(solus.iloc[0, 0])
    # N. of clusters by solution
    ns_clusters = solus.apply(lambda x: x.unique().shape[0])
    # Pic reference solution as one w/ max N. of clusters
    ref = ns_clusters[ns_clusters == ns_clusters.max()].iloc[[0]].index[0]
    # Obtain centroids of reference solution
    ref_centroids = (
        xys.groupby(solus[ref])[xy]
        .apply(lambda xys: xys.mean())
        .drop(lbl_type(-1), errors="ignore")
    )
    # Only continue if any solution
    if ref_centroids.shape[0] > 0:
        # Build KDTree and setup results holder
        ref_kdt = cKDTree(ref_centroids)
        remapped_solus = pd.DataFrame(
            np.zeros(solus.shape, dtype=str), index=solus.index, columns=solus.columns
        )
        if multi_cpu is True:
            import multiprocessing as mp

            pool = mp.Pool(mp.cpu_count())
            s_ids = solus.drop(ref, axis=1).columns.tolist()
            to_loop_over = [(solus[s], ref_centroids, ref_kdt, xys, xy) for s in s_ids]
            remapped = pool.map(_remap_n_expand, to_loop_over)
            remapped_df = pd.concat(remapped, axis=1)
            remapped_solus.loc[:, s_ids] = remapped_df
        else:
            for s in solus.drop(ref, axis=1):
                # -
                pars = (solus[s], ref_centroids, ref_kdt, xys, xy)
                remap_ids = _remap_lbls_single(pars)
                # -
                remapped_solus.loc[:, s] = solus[s].map(remap_ids)
        remapped_solus.loc[:, ref] = solus.loc[:, ref]
        return remapped_solus.fillna("-1")
    else:
        print("WARNING: No clusters identified")
        return solus


def _remap_n_expand(pars):
    solus_s, ref_centroids, ref_kdt, xys, xy = pars
    remap_ids = _remap_lbls_single(pars)
    expanded = solus_s.map(remap_ids)
    return expanded


def _remap_lbls_single(pars):
    new_lbls, ref_centroids, ref_kdt, xys, xy = pars
    lbl_type = type(new_lbls.iloc[0])
    # Cross-walk to cluster IDs
    ref_centroids_ids = pd.Series(ref_centroids.index.values)
    # Centroids for new solution
    solu_centroids = (
        xys.groupby(new_lbls)[xy]
        .apply(lambda xys: xys.mean())
        .drop(lbl_type(-1), errors="ignore")
    )
    # Remapping from old to new labels
    _, nrst_ref_cl = ref_kdt.query(solu_centroids.values)
    remap_ids = pd.Series(nrst_ref_cl, index=solu_centroids.index).map(
        ref_centroids_ids
    )
    return remap_ids


def ensemble(solus, xys, xy=["X", "Y"], multi_cpu=False):
    """
    Generate unique class prediction based on majority/hard voting
    ...

    Arguments
    ---------
    solus       : DataFrame
                  Table with labels for each point (row) and solution (column)
    Returns
    -------
    pred        : DataFrame
                  Table with predictions (`pred`) and proportion of votes 
                  that elected it (`pct`)
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    multi_cpu   : Boolean
                  [Default=False] If True, paralelise remapping
    """
    f = lambda a: Counter(a).most_common(1)[0]
    remapped_solus = remap_lbls(solus, xys, xy=xy, multi_cpu=multi_cpu)
    counts = np.array(list(map(f, remapped_solus.values)))
    winner = counts[:, 0]
    votes = counts[:, 1].astype(int) / solus.shape[1]
    pred = pd.DataFrame({"lbls": winner, "pct": votes})
    return pred
