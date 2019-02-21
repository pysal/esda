import numpy as np
from scipy.sparse import csgraph as cg
from scipy import sparse as sp

try:
    import sklearn.metrics as sk
    import sklearn.metrics.pairwise as skp
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    HAS_REQUIREMENTS = True
except ImportError as e:
    HAS_REQUIREMENTS = False

def _raise_initial_error():
    missing = []
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    raise ImportError("this function requires scikit-learn and "
                      "pandas to be installed. Missing {}".format(','.join(missing)))

def path_silhouette(data, labels, W, D=None, 
                    metric=skp.euclidean_distances, 
                    closest=False, return_nbfc=False, 
                    return_nbfc_score=False,
                    return_paths=False,
                    directed=False):
    """
    Compute a path silhouette for all observations. 

    Parameters
    -----------
    data    :   np.ndarray (N,P)
                matrix of data with N observations and P covariates. 
    labels  :   np.ndarray (N,)
                flat vector of the L labels assigned over N observations. 
    W       :   pysal.W object
                spatial weights object reflecting the spatial connectivity 
                in the problem under analysis
    D       :   np.ndarray (N,N)
                a precomputed distance matrix to apply over W. If passed, 
                takes precedence over data, and data is ignored. 
    metric  :   callable
                function mapping the (N,P) data into an (N,N) dissimilarity matrix, 
                like that found in scikit.metrics.pairwise or scipy.spatial.distance
    closest :   bool
                whether or not to consider the observation "connected" when it 
                is first connected to the cluster, or considering the path cost
                to transit through the cluster. If True, the path cost is assessed
                between i and the path-closest j in each cluster. If False, the path
                cost is assessed as the average of path costs between i and all j
                in each cluster
    return_nbfc     :   bool
                        Whether or not to return the label of the next best fit 
                        cluster
    return_nbfc_score:  bool
                        Whether or not to return the score of the next best fit 
                        cluster. 
    return_paths    :   bool
                        Whether or not to return the matrix of shortest path 
                        lengths after having computed them. 
    directed    :   bool
                    whether to consider the weights matrix as directed or undirected.
                    If directed, asymmetry in the input W is heeded. If not,
                    asymmetry is ignored. 

    Returns
    --------
    An (N,) array of the clusters involved. 
    """
    if not HAS_REQUIREMENTS:
        _raise_initial_error()

    if D is None:
        D = metric(data)
    #polymorphic for sparse & dense input
    assert 0 == (D < 0).sum(), "distance metric returns negative values"
    off_diag_zeros = (D + np.eye(D.shape[0])) == 0
    D[off_diag_zeros] = -1
    Wm = sp.csr_matrix(W.sparse)
    DW = sp.csr_matrix(Wm.multiply(D))
    DW.eliminate_zeros()
    DW[DW < 0] = 0
    assert 0 == (DW < 0).sum()
    all_pairs = cg.shortest_path(DW, directed=directed)
    labels = np.asarray(labels)
    if W.n_components > 1:
        from libpysal.weights.util import WSP
        psils_ = np.empty(W.n, dtype=float)
        closest_connecting_label_ = np.empty(W.n, dtype=labels.dtype)
        closest_connection_score_ = np.empty(W.n, dtype=labels.dtype)
        for component in np.unique(W.component_labels):
            this_component_mask = np.nonzero(W.component_labels == component)[0]
            subgraph = W.sparse[this_component_mask.reshape(-1,1), # these rows
                                this_component_mask.reshape(1,-1)] #these columns
            subgraph_W = WSP(subgraph).to_W()
            assert subgraph_W.n_components == 1
            # DW operation is idempotent
            subgraph_D = DW[this_component_mask.reshape(-1,1), # these rows
                            this_component_mask.reshape(1,-1)] #these columns
            subgraph_labels = labels[this_component_mask]
            n_subgraph_labels = len(np.unique(subgraph_labels))
            if not (2 < n_subgraph_labels < (subgraph_W.n-1)) :
                psils = subgraph_solutions = [0]*subgraph_W.n
                closest_connecting_label = [np.nan]*subgraph_W.n
                closest_connection_score = [np.inf]*subgraph_W.n
            else:
                subgraph_solutions = path_silhouette(data=None, labels=subgraph_labels, W=subgraph_W, D=subgraph_D, 
                                                     metric=metric, closest=closest, return_nbfc=return_nbfc, 
                                                     return_nbfc_score=return_nbfc_score,
                                                     return_paths=return_paths,
                                                     directed=directed)
            # always throw away all_pairs, since we already have it built
                if (return_nbfc or return_nbfc_score) and return_paths:
                    if return_nbfc_score:
                        (psils, closest_connecting_label, 
                                closest_connection_score,
                                      _) = subgraph_solutions
                    else:
                        psils, closest_connecting_label, _ = subgraph_solutions
                elif return_nbfc_score:
                    psils, closest_connecting_label, closest_connection_score = subgraph_solutions
                elif return_nbfc:
                    psils, closest_connecting_label = subgraph_solutions
                elif return_paths:
                    psils, _ = subgraph_solutions
                else:
                    psils = subgraph_solutions
            if return_nbfc:
                closest_connecting_label_[this_component_mask] = closest_connecting_label
            if return_nbfc_score:
                closest_connection_score_[this_component_mask] = closest_connection_score
            psils_[this_component_mask] = psils
        closest_connection_score = closest_connection_score_
        closest_connecting_label = closest_connecting_label_
        psils = psils_
    # Single Connected Component
    elif closest is False:
        psils = sk.silhouette_samples(all_pairs, labels, metric='precomputed')
        if return_nbfc or return_nbfc_score:
            closest_connecting_label = []
            closest_connection_score = []
            for i, label in enumerate(labels):
                row = all_pairs[i].copy()
                in_label = labels == label
                masked_label = row.copy() # for observations in the row
                masked_label[in_label] = np.inf # make those in cluster infinite
                nearest_not_in_cluster = np.argmin(masked_label) # find the closest
                nearest_not_in_cluster_label = labels[nearest_not_in_cluster] #label
                nearest_not_in_cluster_score = masked_label[nearest_not_in_cluster]
                closest_connecting_label.append(nearest_not_in_cluster_label)
                closest_connection_score.append(nearest_not_in_cluster_score)
    else:
        psils = []
        closest_connecting_label = []
        closest_connection_score = []
        for i,label in enumerate(labels):
            row = all_pairs[i]
            in_label = labels == label
            #required to make argmin pertain to N, not N - len(in_label)
            masked_label = row.copy()
            masked_label[in_label] = np.inf
            nearest_not_in_cluster = np.argmin(masked_label)
            nearest_not_in_cluster_score = row[nearest_not_in_cluster]
            nearest_not_in_cluster_label = labels[nearest_not_in_cluster]

            average_interconnect_in_cluster = row[in_label].mean()
            psil = nearest_not_in_cluster_score - average_interconnect_in_cluster
            psil /= np.maximum(nearest_not_in_cluster_score, average_interconnect_in_cluster)
            psils.append(psil)
            closest_connecting_label.append(nearest_not_in_cluster_label)
            closest_connection_score.append(nearest_not_in_cluster_score)
        psils = np.asarray(psils)
    if (return_nbfc or return_nbfc_score) and return_paths:
        if return_nbfc_score:
            out = (psils, np.asarray(closest_connecting_label), 
                          np.asarray(closest_connection_score),
                          all_pairs )
        else:
            out = psils, np.asarray(closest_connecting_label), all_pairs
    elif return_nbfc_score:
        out = psils, np.asarray(closest_connecting_label),\
                     np.asarray(closest_connection_score)
    elif return_nbfc:
        out = psils, np.asarray(closest_connecting_label)
    elif return_paths:
        out = psils, all_pairs
    else:
        out = psils
    return out

def boundary_silhouette(data, labels, W, metric=skp.euclidean_distances):
    """
    Compute the observation-level boundary silhouette score. 

    The boundary silhouette is the silhouette score using only spatially-proximate
    clusters as candidates for the next-best-fit distance function (the 
    b(i) function in Rosseeuw (1987)). 
    This restricts the next-best-fit cluster to be the set of clusters on which 
    an observation neighbors. 
    So, instead of considering *all* clusters when finding the next-best-fit cluster, 
    only clusters that `i` borders are considered. 
    This is supposed to model the fact that, in spatially-constrained clustering, 
    observation i can only be reassigned from cluster c to cluster k if some observation 
    j neighbors i and also resides in k. 

    If an observation only neighbors its own cluster, i.e. is not on the boundary
     of a cluster, this value is zero. 

    If a cluster has exactly one observation, this value is zero. 

    If an observation is on the boundary of more than one cluster, then the
    best candidate is chosen from the set of clusters on which the observation borders.  

    metric is a callable mapping an (N,P) data into an (N,N) distance matrix OR
    an (N,N) distance matrix already. 
    """
    if not HAS_REQUIREMENTS:
        _raise_initial_error()

    alist = W.to_adjlist()
    labels = np.asarray(labels)
    if callable(metric):
        full_distances = metric(data)
    elif isinstance(metric, np.ndarray):
        if metric.shape == (n_obs, n_obs):
            full_distances = metric
    label_frame = pd.DataFrame(labels, index=W.id_order, columns=['label'])
    alist = alist.merge(label_frame, left_on='focal', right_index=True, how='left')\
                 .merge(label_frame, left_on='neighbor', right_index=True, how='left',
                        suffixes=('_focal', '_neighbor'))
    alist['boundary'] = alist.label_focal != alist.label_neighbor
    focals = alist.groupby('focal')
    bmask = focals.boundary.any()
    result = []
    np.seterr(all='raise')
    try:
        for i, (ix,bnd) in enumerate(bmask.iteritems()):
            if not bnd:
                result.append(np.array([0]))
                continue
            sil_score = np.array([np.inf])
            label = labels[i]
            focal_mask = np.nonzero(labels == label)[0]
            if len(focal_mask) == 1: #the candidate is singleton
                result.append(np.array([0]))
                continue
            neighbors = alist.query("focal == {}".format(ix)).label_neighbor
            mean_dissim = full_distances[i,focal_mask].sum() / (len(focal_mask)-1)
            if not np.isfinite(mean_dissim).all():
                break_reason = 'a non-finite mean dissimilarity'
                raise
            neighbor_score = np.array([np.inf])
            for neighbor in set(neighbors).difference([label]):
                other_mask = np.nonzero(labels == neighbor)[0]
                other_score = full_distances[i,other_mask].mean()
                neighbor_score = np.minimum(neighbor_score, other_score, neighbor_score)
                if neighbor_score < 0:
                    break_reason = 'negative neighbors'
                    raise
            sil_score = (neighbor_score - mean_dissim) / np.maximum(neighbor_score, 
                                                                    mean_dissim)
            result.append(sil_score)
        if len(result) != len(labels):
            break_reason = 'mismatch in iteration sizes'
            raise
    except:
        return locals()
    return np.asarray(result)

def silhouette_alist(data, labels, alist, indices=None,
                     metric=skp.euclidean_distances):
    """
    Compute the silhouette for each edge in an adjacency graph. Given the alist 
    containing `focal` id, `neighbor` id, and `label_focal`, and `label_neighbor`, 
    this computes:

    `d(i,label_neighbor) - d(i,label_focal) / (max(d(i,label_neighbor), d(i,label_focal)))`

    data : (N,P) matrix to cluster on or DataFrame indexed on the same values as
           that in alist.focal/alist.neighbor
    labels: (N,) Series containing classifications, indexed on the same values 
                 as that in alist.focal/alist.neighbor
    alist: adjacency list containing columns focal & neighbor, 
           describing one edge of the graph. 
    indices: (N,) array containing the "name" for observations in
           alist to be linked to data. indices should be:
           1. aligned with data by iteration order
           2. include all values in the alist.focal set. 
           if alist.focal and alist.neighbor are strings, then indices should be
           a list/array of strings aligned with the rows of data. 
           if not provided and labels is a series/dataframe, 
           then its index will be used. 
    """
    if not HAS_REQUIREMENTS:
        _raise_initial_error()

    n_obs = data.shape[0]
    if callable(metric):
        full_distances = metric(data)
    elif isinstance(metric, np.ndarray):
        if metric.shape == (n_obs, n_obs):
            full_distances = metric
    if isinstance(data, pd.DataFrame):
        indices = data.index
    if isinstance(labels, (pd.DataFrame, pd.Series)) and indices is None:
        indices = labels.index
    elif indices is not None \
        and not isinstance(labels, (pd.DataFrame, pd.Series)):
        labels = pd.Series(labels, index=indices)
    elif indices is None \
        and not isinstance(labels, (pd.DataFrame, pd.Series)):
        indices = np.arange(len(labels))
        labels = pd.Series(labels, index=indices)
    if isinstance(labels, pd.DataFrame):
        labels = pd.Series(labels.values, index=labels.index)
    assert indices is not None
    assert isinstance(labels, pd.Series)
    labels = labels.to_frame("label")

    result = alist.sort_values('focal').copy(deep=True)

    result = result.merge(labels, left_on='focal', right_index=True, how='left')\
                   .merge(labels, left_on='neighbor', right_index=True, how='left',
                          suffixes=('_focal','_neighbor'))
    self_dcache = dict()
    sils = []
    indices = list(indices)
    for i_alist, row in result.iterrows():
        name = row.focal
        label = row.label_focal
        neighbor_label = row.label_neighbor
        if neighbor_label == label:
            sils.append(0)
            continue
        i_Xc = indices.index(name)
        mask = labels == label
        mask = np.nonzero(mask.values)[0]
        within_cluster = self_dcache.get((i_Xc,label), full_distances[i_Xc,mask].mean())
        self_dcache[(i_Xc,label)] = within_cluster
        neighbor_mask = labels == neighbor_label
        neighbor_mask = np.nonzero(neighbor_mask.values)[0]
        if len(neighbor_mask) == 0:
            print(neighbor_label)
            raise
        outer_distance = full_distances[i_Xc,neighbor_mask].mean()
        sils.append((outer_distance - within_cluster) / np.maximum(outer_distance, within_cluster))
    result['silhouette'] = sils
    return result.sort_values("focal").reset_index(drop=True)

def nearest_label(data, labels, 
                  metric=skp.euclidean_distances, 
                  return_distance=False, keep_self=False):
    """
    Find the nearest label in attribute space. 

    Given the data and a set of labels in labels, this finds the label 
    whose mean center is closest to the observation in data. 
    """
    if not HAS_REQUIREMENTS:
        _raise_initial_error()

    if callable(metric):
        dissim = metric(data)
    elif metric.lower == "precomputed":
        assert data.shape == (labels.shape[0],labels.shape[0]), "dissimilarity matrix is malformed!"
        dissim = data
    elif isinstance(metric, np.ndarray):
        assert metric.shape == (labels.shape[0],labels.shape[0]), "dissimilarity matrix is malformed!"
        dissim = metric
    unique_labels = np.unique(labels)
    nearest_label = np.empty(labels.shape, dtype=labels.dtype)
    nearest_label_dissim = np.empty(labels.shape)
    for label in unique_labels:
        this_label_mask = labels == label
        n_in_label = this_label_mask.sum()
        this_label_mask = np.nonzero(this_label_mask)[0]
        next_best_fit = np.ones(this_label_mask.shape) * np.inf
        next_best_label = np.empty(this_label_mask.shape, dtype=labels.dtype)
        for neighbor in unique_labels:
            if (neighbor == label) & (not keep_self):
                continue
            neighbor_label_mask = labels == neighbor
            n_in_neighbor = neighbor_label_mask.sum()
            neighbor_label_mask = np.nonzero(neighbor_label_mask)[0].reshape(1,-1)
            # Need to account for the fact that the self-distance
            # is not included in the silhouette; in small clusters, 
            # this extra zero can bring down the average, resulting in a case
            # where the silhouette is negative, but the "nearest" cluster would
            # be the current cluster if we take averages including i in C. 
            chunk = dissim[this_label_mask.reshape(-1,1), #these rows
                           neighbor_label_mask] #and these columns
            neighbor_distance = chunk.sum(axis=1) / np.maximum(n_in_neighbor-1, 1) #and sum across rows
            next_best_label[neighbor_distance < next_best_fit] = neighbor
            np.minimum(next_best_fit, neighbor_distance, next_best_fit)
        nearest_label[this_label_mask] = next_best_label
        nearest_label_dissim[this_label_mask] = next_best_fit
    if return_distance:
        return nearest_label, nearest_label_dissim
    else:
        return nearest_label
