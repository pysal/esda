import numpy
from scipy import spatial
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array
from scipy.sparse import csgraph
from scipy.stats import mode as most_common_value
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def _resolve_metric(metric):
    if callable(metric):
        distance_func = metric
    elif metric.lower() == 'haversine':
        try:
            from numba import autojit
        except:
            def autojit(func):
                return func
        @autojit
        def harcdist(p1,p2):
            """ Compute the kernel of """
            x = numpy.sin(p2[1] - p1[1]/2)**2 
            y = numpy.cos(p2[1])*numpy.cos(p1[1]) * numpy.sin((p2[0] - p1[0])/2)**2
            return 2 * numpy.arcsin(numpy.sqrt(x + y))
        distance_func = harcdist
    elif metric.lower() == 'precomputed':
        distances = check_array(coordinates, accept_sparse=True)
        assert distances.shape == (n,n), ('With metric="precomputed", distances'
                                          ' must be an (n,n) matrix.')
        raise NotImplementedError()
        # would need to:
        # get index of point pairs
        # get distance corresponding to that point pair
    else:
        try:
            distance_func = getattr(distance, metric)
        except AttributeError:
            raise KeyError('Metric {} not understood. Choose something available in scipy.spatial.distance'.format(metric))
    return distance_func


def isolation(X, coordinates, metric='euclidean', return_all=False):
    X = check_array(X)
    n, p = X.shape
    assert p == 1, 'isolation is a univariate statistic'
    X = X.flatten()
    try:
        from rtree.index import Index as SpatialIndex
    except ImportError:
        raise ImportError('rtree library must be installed to use '
                          'the prominence measure')
    distance_func = _resolve_metric(metric)
    sort_order = numpy.argsort(-X)
    tree = SpatialIndex()
    tree.insert(0, tuple(coordinates[sort_order][0]), obj=X[sort_order][0])
    ix = numpy.where(sort_order == 0)[0].item()
    precedence_tree = [[ix, numpy.nan, 0, numpy.nan, numpy.nan, numpy.nan]]
    for i, (value, location) in enumerate(zip(X[sort_order][1:], 
                                                 coordinates[sort_order][1:])):
        rank = i + 1
        ix = numpy.where(sort_order == rank)[0].item()
        match, = tree.nearest(tuple(location), objects=True)
        higher_rank = match.id
        higher_value = match.object
        higher_location = match.bbox[:2]
        higher_ix = numpy.where(sort_order == higher_rank)[0].item()
        distance = distance_func(location, higher_location)
        gap = higher_value - value
        precedence_tree.append([ix, higher_ix, rank, higher_rank, 
                                distance, gap])
        tree.insert(rank, tuple(location), obj=value)
    #return precedence_tree
    precedence_tree = numpy.asarray(precedence_tree)
    #print(precedence_tree.shape)
    out = numpy.empty_like(precedence_tree)
    out[sort_order] = precedence_tree
    isolation = pandas.DataFrame(out, columns = ['index', 'parent_index',
                                                 'rank', 'parent_rank',
                                                 'distance', 'gap'])
    if return_all:
        return isolation
    else:
        return isolation.distance.values

def prominence(X, connectivity, return_saddles=False, 
               return_peaks=False, return_dominating_peak=False, gdf=None,
               verbose=False):
        raise Exception('need to preprocess the peaks so that you only call something a keycol if it joins *new* unjoined peaks!')
        X = check_array(X.squeeze(), ensure_2d=False)
        
        assert len(X.shape) == 1, 'prominence is a univariate statistic'
        n, = X.shape

        # sort the variable in ascending order
        sort_order = numpy.argsort(-X)

        last_n = 0
        peaks = [sort_order[0]]
        assessed_peaks = set()
        prominence = numpy.empty_like(X)*numpy.nan
        dominating_peak = numpy.ones_like(X)*-1
        predecessors = numpy.ones_like(X)*-1
        key_cols = dict()
        for rank, value in tqdm(enumerate(X[sort_order])):
            # This is needed to break ties in the same way that argsort does. A more
            # natural way to do this is to use X >= value, but if value is tied, then
            # that would generate a mask where too many elements are selected! 
            # e.g. mask.sum() > rank
            mask = numpy.isin(numpy.arange(n), sort_order[:rank+1])
            full_indices, = mask.nonzero()
            this_full_ix = sort_order[rank]
            msg = 'assessing {} (rank: {}, value: {})'.format(this_full_ix, rank, value)
            # This is again needed to break ties in the same way that argsort does.
            # Basically, you can never rely on using the `value` and ought always use the
            # `rank` or argsort item. It maps from the evaluated set to the full set of values
            this_reduced_ix = full_indices.tolist().index(this_full_ix)
            # This is the subgraph of all places that have been classified as peak/kc/slope
            
            ## to make this faster:
            # use the dominating_peak vector. A new obs either has:
            # 1. neighbors whose dominating_peak are all -1 (new peak)
            # 2. neighbors whose dominating_peak are all -1 or an integer (slope of current peak)
            # 3. neighbors whose dominating_peak include at least two integers and any -1 (key col)
            print(this_full_ix)
            _,neighbs = connectivity[this_full_ix,].toarray().nonzero()
            this_preds = predecessors[neighbs]
            # want to keep ordering in this sublist to preserve hierarchy
            this_unique_preds = [p for p in peaks 
                                if ((p in this_preds)
                                    & (p >= 0))]
            if tuple(this_unique_preds) in key_cols.keys():
                classification = 'slope'
            elif len(this_unique_preds) == 0:
                classification = 'peak'
            elif len(this_unique_preds) >= 2:
                classification = 'keycol'
            else:
                classification = 'slope'
                
            #subgraph = connectivity[full_indices, full_indices.reshape(-1,1)]
            #this_n, reduced_labels = csgraph.connected_components(subgraph)
            # since the subgraph is over `rank` observations, we need to get both the label
            #this_label = reduced_labels[this_reduced_ix]
            # and the "original label" in the full `n` observation set. 
            #full_labels = numpy.ones_like(mask) * -1
            #full_labels[mask] = reduced_labels
            # This gives us the indices (in the full `n` set) of elements in 
            # the subgraph's own graph component. 
            #full_in_this_label_ix = set(numpy.where(full_labels == this_label)[0])
            if classification == 'keycol': # this_ix merges two or more subgraphs, so is a key_col
                # find the peaks it joins
                #now_joined_peaks = [p for p in peaks if p in full_in_this_label_ix]
                now_joined_peaks = this_unique_preds
                #print(now_joined_peaks, this_unique_preds)
                #numpy.testing.assert_equal(now_joined_peaks, this_unique_preds)
                # add them as keys for the key_col lut
                key_cols.update({tuple(now_joined_peaks):this_full_ix})
                msg += '\n{} is a key col between {}!'.format(this_full_ix, now_joined_peaks)
                dominating_peak[this_full_ix] = now_joined_peaks[-1] # lowest now-joined peak
                predecessors[this_full_ix] = now_joined_peaks[-1]
                prominence[this_full_ix] = 0
                # given we now know the key col, get the prominence for 
                # unassayed peaks in the subgraph
                for peak_ix in now_joined_peaks:
                    if peak_ix in assessed_peaks:
                        continue
                    # prominence is peak - key col
                    prominence[peak_ix] -= value
                    assessed_peaks.update((peak_ix,))
            elif classification == 'peak': # this_ix is a new peak since it's disconnected
                msg += '\n{} is a peak!'.format(this_full_ix)
                # its parent is the last visited peak (for precedence purposes)
                previous_peak = peaks[-1]
                if not (this_full_ix in peaks):
                    peaks.append(this_full_ix)
                dominating_peak[this_full_ix] = previous_peak
                predecessors[this_full_ix] = this_full_ix
                # we initialize prominence here, rather than compute it solely in
                # the `key_col` branch because a graph `island` disconnected observation
                # should have prominence "value - 0", since it has no key cols
                prominence[this_full_ix] = X[this_full_ix]
            else: # this_ix is connected to an existing peak, but doesn't bridge peaks.
                msg += '\n{} is a slope!'.format(this_full_ix)
                # get all the peaks that are linked to this slope
                this_peak = this_unique_preds
                if len(this_peak) == 1: # if there's only one peak the slope is related to
                    # then use it
                    best_peak = this_peak[0]
                else: # otherwise, if there are multiple peaks
                    # pick the one that most of its neighbors are assigned to
                    best_peak = most_common_value(this_unique_preds).mode.item()
                all_on_slope = numpy.arange(n)[dominating_peak == best_peak]
                msg += '\n{} are on the slopes of {}'.format(all_on_slope, best_peak)
                dominating_peak[this_full_ix] = best_peak
                predecessors[this_full_ix] = best_peak
            if verbose:
                print('--------------------------------------------\n'
                      'at the {} iteration:\n{}\n\tpeaks\t{}\n\tprominence\t{}\n\tkey_cols\t{}\n'
                      ''.format(rank, msg, peaks, prominence, key_cols))
            if gdf is not None:
                peakframe = gdf.iloc[peaks]
                keycolframe = gdf.iloc[list(key_cols.values())]
                slopeframe = gdf[(~(gdf.index.isin(peakframe.index)
                                    | gdf.index.isin(keycolframe.index)))
                                 & mask]
                rest = gdf[~mask]
                ax = rest.plot(edgecolor='k', linewidth=.1, facecolor='lightblue')
                ax = slopeframe.plot(edgecolor='k', linewidth=.1, facecolor='linen', ax=ax)
                ax = keycolframe.plot(edgecolor='k', linewidth=.1, facecolor='red', ax=ax)
                ax = peakframe.plot(edgecolor='k', linewidth=.1, facecolor='yellow', ax=ax)
                plt.show()
                command = input()
        if not any((return_saddles, return_peaks, return_dominating_peak)):
            return prominence
        retval = [prominence]
        if return_saddles:
            retval.append(key_cols)
        if return_dominating_peak:
            retval.append(dominating_peak)
        return retval
        

if __name__ == '__main__':
    import geopandas, pandas, numpy
    from libpysal import weights
    data = geopandas.read_file('../cb_2015_us_county_500k_2.geojson')
    contig = data.query('statefp not in ("02", "15", "43", "72")').reset_index()
    coordinates = numpy.column_stack((contig.centroid.x, contig.centroid.y))
    income = contig[['median_income']].values.flatten()
    contig_graph = weights.Rook.from_dataframe(contig)
    #iso = isolation(income, coordinates, return_all=True)
    #contig.assign(isolation = iso.distance.values).plot('isolation')
    
    wa = contig.query('statefp == "53"').reset_index()
    wa_income = wa[['median_income']].values
    wa_graph = weights.Rook.from_dataframe(wa)
    
    
    ca = contig.query('statefp == "06"').reset_index()
    ca_income = ca[['median_income']].values
    ca_graph = weights.Rook.from_dataframe(ca)