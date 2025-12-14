import numpy as np
from .moran import Moran_Local
from .significance import calculate_significance
from libpysal.weights import lag_spatial
from libpysal.graph import Graph

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def _calc_quad(x,y):
    """
    This is a simpler solution to calculate a cartesian quadrant. 

    To explain graphically, let the tuple below be (off_sign[i], neg_y[i]*2). 

    If sign(x[i]) != sign(y[i]), we are on the negative diagonal. 
    If y is negative, we are on the bottom of the plot. 

    Therefore, the sum (off_sign + neg_y*2 + 1) gives you the cartesian quadrant. 

     II  |   I
     1,0 | 0,0
    -----+-----
     0,2 | 1,2
     III |  IV

    """
    off_sign = np.sign(x) != np.sign(y)
    neg_y = (y<0)
    return off_sign + neg_y*2 + 1

class MoranLocalPartial(object):
    def __init__(
        self, permutations=999, unit_scale=True, partial_labels=True, alternative='two-sided'
    ):
        """
        Compute the Multivariable Local Moran statistics under partial dependence :cite:`wolf2024confounded`

        Parameters
        ---------
        permutations : int
            the number of permutations to run for the inference,
            driven by conditional randomization.
        unit_scale : bool
            whether to enforce unit variance in the local statistics. This
            normalizes the variance of the data at inupt, ensuring that
            the covariance statistics are not overwhelmed by any single
            covariate's large variance.
        partial_labels : bool, default=True
            whether to calculate the classification based on the part-regressive 
            quadrant classification or the univariate quadrant classification,
            like a classical Moran's I. When mvquads is True, the variables are labelled as:
            - label 1: observations with large y - rho * x that also have large Wy values. 
            - label 2: observations with small y - rho * x values that also have large Wy values.
            - label 3: observations with small y - rho * x values that also have small Wy values. 
            - label 4: observations with large y - rho * x values that have small Wy values.
        alternative : str (default: 'two-sided')
            the alternative hypothesis for the inference. One of
            'two-sided', 'greater', 'lesser', 'directed', or 'folded'.
            See the esda.significance.calculate_significance() documentation
            for more information.

        Attributes
        ----------
        connectivity : The weights matrix inputted, but row standardized
        D : The "design" matrix used in computation. If X is
            not None, this will be [1 y X]
        R : the "response" matrix used in computation. Will
            always be the same shape as D and contain [1, Wy, Wy, ....]
        DtDi : empirical parameter covariance matrix
            the P x P matrix describing the variance and covariance
            of y and X.
        P : the number of parameters. 1 if X is not provided.
        association_ : the N,P matrix of multivariable LISA statistics.
            the first column, lmos[:,1] is the LISAs corresponding
            to the relationship between Wy and y conditioning on X.
        reference_distribution_ : the (N, permutations, P+1) realizations from the conditional
            randomization to generate reference distributions for
            each Local Moran statistic. rlmos_[:,:,1] pertain to
            the reference distribution of y and Wy.
        significance_  : the (N, P) matrix of quadrant classifications for the
            part-regressive relationships. quads[:,0] pertains to
            the relationship between y and Wy. The mean is not classified,
            since it's just binary above/below mean usually.
        partials_: the (N,2,P+1) matrix of part-regressive contributions.
            The ith slice of partials_[:,:,i] contains the
            partial regressive contribution of that covariate, with
            the first column indicating the part-regressive outcome
            and the second indicating the part-regressive design.
            The partial regression matrix starts at zero, so
            partials_[:,:,0] corresponds to the partial regression
            describing the relationship between y and Wy.
        labels_ : the (N,) array of quadrant classifications for the
            part-regressive relationships. See the partial_labels argument
            for more information. 
        """
        self.permutations = permutations
        self.unit_scale = unit_scale
        self.partial_labels = partial_labels
        self.alternative = alternative
    
    def fit(self, X, y, W):
        """
        Fit the partial local Moran statistic on input data

        Parameters
        ----------
        X   : (N,p) array
            array of data that is used as "confounding factors"
            to account for their covariance with Y.
        y   : (N,1) array
            array of data that is the targeted "outcome" covariate
            to compute the multivariable Moran's I
        W   : (N,N) weights object
            spatial weights instance as W or Graph aligned with y. Immediately row-standardized.
      
        Returns
        -------
        self    :   object
            this MoranLocalPartial() statistic after fitting to data
        """
        y = np.asarray(y).reshape(-1, 1)
        if isinstance(W, Graph):
            W = W.transform("R")
        else:
            W.transform = "r"
        y = y - y.mean()
        if self.unit_scale:
            y /= y.std()
        X = X - X.mean(axis=0)
        if self.unit_scale:
            X = X / X.std(axis=0)
        self.y = y
        self.X = X
        D, R = self._make_data(y, X, W)
        self.D, self.R = D, R
        self.P = D.shape[1] - 1
        self.N = W.n
        self.DtDi = np.linalg.inv(
            self.D.T @ self.D
        )  # this is only PxP, so not too bad...
        self._left_component_ = (self.D @ self.DtDi) * (self.N - 1)
        self._lmos_ = self._left_component_ * self.R
        self.connectivity = W
        self.permutations = self.permutations
        if self.permutations is not None:  # NOQA necessary to avoid None > 0
            if self.permutations > 0:
                self._crand(y, X, W)
                self._rlmos_ *= self.N - 1
                self._p_sim_ = np.zeros((self.N, self.P + 1))
                for i in range(self.P + 1):
                    self._p_sim_[:,i] = calculate_significance(
                        self._lmos_[:,i], 
                        self._rlmos_[:,:,i], 
                        alternative=self.alternative
                    )

        component_quads = []
        for i, left in enumerate(self._left_component_.T):
            right = self.R[:, i]
            quads = _calc_quad(left - left.mean(), right)
            component_quads.append(quads)
        self._partials_ = np.asarray(
            [
                np.vstack((left, right)).T
                for left, right in zip(self._left_component_.T, self.R.T)
            ]
        )

        uvquads = []
        negative_lag = R[:,1] < 0
        for i, x_ in enumerate(self.D.T):
            if i == 0:
                continue
            off_sign = np.sign(x_) != np.sign(R[:,1])
            quads = negative_lag.astype(int).flatten() * 2 + off_sign.astype(int) + 1
            uvquads.append(quads.flatten())

        self._uvquads_ = np.row_stack(uvquads).T
        self._mvquads_ = np.row_stack(component_quads).T
        return self

    def _make_data(self, z, X, W):
        if isinstance(W, Graph): # NOQA because ternary is confusing
            Wz = W.lag(z)
        else:
            Wz = lag_spatial(W, z)
        if X is not None:
            D = np.hstack((np.ones(z.shape), z, X))
            P = X.shape[1] + 1
        else:
            D = np.hstack((np.ones(z.shape), z))
            P = 1
        R = np.tile(Wz, P + 1)
        return D, R
        # self.D, self.R = D, R

    def _crand(self, y, X, W):
        N = W.n
        N_permutations = self.permutations
        prange = range(N_permutations)
        if isinstance(W, Graph):
            max_neighbs = W.cardinalities.max() + 1
        else:
            max_neighbs = W.max_neighbors + 1
        pre_permutations = np.array(
            [np.random.permutation(N - 1)[0:max_neighbs] for i in prange]
        )
        straight_ids = np.arange(N)
        if isinstance(W, Graph): # NOQA
            id_order = W.unique_ids
        else:
            id_order = W.id_order
        DtDi = self.DtDi
        ordered_weights = [W.weights[id_order[i]] for i in straight_ids]
        ordered_cardinalities = [W.cardinalities[id_order[i]] for i in straight_ids]
        lmos = np.empty((N, N_permutations, self.P + 1))
        for i in tqdm(range(N), desc="Simulating by site"):
            ids_noti = straight_ids[straight_ids != i]
            np.random.shuffle(ids_noti)
            these_permutations = pre_permutations[:, 0 : ordered_cardinalities[i]]
            randomized_permutations = ids_noti[these_permutations]
            shuffled_ys = y[randomized_permutations]
            these_weights = np.asarray(ordered_weights[i]).reshape(-1, 1)
            shuffled_Wyi = (shuffled_ys * these_weights).sum(
                axis=1
            )  # these are N-permutations by 1 now
            # shuffled_X = X[randomized_permutations, :] 
            # #these are still N-permutations, N-neighbs, N-covariates
            if X is None:
                local_data = np.array((1, y[i].item())).reshape(1, -1)
                shuffled_D = np.tile(
                    local_data, N_permutations
                ).T  # now N permutations by P
            else:
                local_data = np.array((1, y[i].item(), *X[i])).reshape(-1, 1)
                shuffled_D = np.tile(
                    local_data, N_permutations
                ).T  # now N permutations by P
            shuffled_R = np.tile(shuffled_Wyi, self.P + 1)
            lmos[i] = (shuffled_R * shuffled_D) @ DtDi
        self._rlmos_ = lmos  # nobs, nperm, nvars

    @property
    def association_(self):
        """
        The association between y and the local average of y,
        removing the correlation due to x and the local average of y
        """
        return self._lmos_[:, 1]

    @property
    def significance_(self):
        """
        The pseudo-p-value built using map randomization for the
        structural relationship between y and its local average,
        removing the correlation due to the relationship between x
        and the local average of y.
        """
        return self._p_sim_[:, 1]

    @property
    def partials_(self):
        """
        The components of the local statistic. The first column is the
        structural exogenous component of the data, and the second is the
        local average of y.
        """
        return self._partials_[1]

    @property
    def reference_distribution_(self):
        """
        Simulated distribution of association_, assuming that there is
          - no structural relationship between y and its local average;
          - the same observed structural relationship between y and x.
        """
        return self._rlmos_[:, :, 1]

    @property
    def labels_(self):
        """
        The classifications (in terms of cluster-type and outlier-type)
        for the association_ statistics. If the quads requested are
        *mvquads*, then the classification is done with respect to the
        left and right components (first and second columns of partials_).

        If the quads requested are *uvquads*, then this will only be computed
        with respect to the outcome and the local average.
        The cluster typology is:
          - 1: above-average left component (either y or D @ DtDi),
               above-average right component (local average of y)
          - 2: below-average left component (either y or D @ DtDi),
               above-average right component (local average of y)
          - 3: below-average left component (either y or D @ DtDi)
               below-average right component (local average of y)
          - 4: above-average left component (either y or D @ DtDi)
               below-average right component (local average of y)
        """
        if self.partial_labels:
            return self._mvquads_[:, 1]
        else:
            return self._uvquads_[:, 1]


class MoranLocalConditional(Moran_Local):
    """
    Fit a local moran statistic for y after regressing out the
    effects of confounding X on y. A "stronger" version of the
    MoranLocalPartial statistic, as defined by :cite:`wolf2024confounded`
    """

    def __init__(
        self,
        permutations=999,
        unit_scale=True,
        transformer=None,
        alternative='two-sided'
    ):
        """
        Initialize a local Moran statistic on the regression residuals

        Parameters
        ---------
        permutations    : int (default: 999)
                          the number of permutations to run for the inference,
                          driven by conditional randomization.
        unit_scale      : bool (default: True)
                          whether or not to convert the input data to a unit normal scale.
        transformer     : callable (default: scikit regression)
                          should transform X into a predicted y. If not provided, will use
                          the standard scikit OLS regression of y on X.
        alternative     : str (default: 'two-sided')
                          the alternative hypothesis for the inference. One of
                          'two-sided', 'greater', 'lesser', 'directed', or 'folded'.
                          See the esda.significance.calculate_significance() documentation
                          for more information.

        Attributes
        ----------
        connectivity : The weights matrix inputted, but row standardized
        y_filtered_ : the (N,1) array of y after removing the effect of X
        association_ : the N,P matrix of multivariable LISA statistics.
            the first column, lmos[:,1] is the LISAs corresponding
            to the relationship between Wy and y conditioning on X.
        reference_distribution_ : the (N, permutations, P+1) realizations from the conditional
            randomization to generate reference distributions for
            each Local Moran statistic. rlmos_[:,:,1] pertain to
            the reference distribution of y and Wy.
        significance_ : the (N, P) matrix of quadrant classifications for the
            part-regressive relationships. quads[:,0] pertains to
            the relationship between y and Wy. The mean is not classified,
            since it's just binary above/below mean usually.
        partials_ : the (N,2,P+1) matrix of part-regressive contributions.
            The ith slice of partials_[:,:,i] contains the
            partial regressive contribution of that covariate, with
            the first column indicating the part-regressive outcome
            and the second indicating the part-regressive design.
            The partial regression matrix starts at zero, so
            partials_[:,:,0] corresponds to the partial regression
            describing the relationship between y and Wy.
        labels_ : the (N,) array of quadrant classifications for the
            part-regressive relationships. See the partial_labels argument
            for more information. 
        """
        self.permutations = permutations
        self.unit_scale = unit_scale
        self.transformer = transformer
        self.alternative = alternative

    def fit(self, X, y, W):
        """
        Parameters
        ---------
        y : (N,1) array
            array of data that is the targeted "outcome" covariate
            to compute the multivariable Moran's I
        X : (N,3) array
            array of data that is used as "confounding factors"
            to account for their covariance with Y.
        W : (N,N) weights object
            spatial weights instance as W or Graph aligned with y. Immediately row-standardized.
        
        Returns
        -------
        A fitted MoranLocalConditional() estimator
        """
        y = y - y.mean()
        X = X - X.mean(axis=0)
        if self.unit_scale:
            y /= y.std()
            X /= X.std(axis=0)
        self.y = y
        self.X = X
        y_filtered_ = self.y_filtered_ = self._part_regress_transform(y, X)
        if isinstance(W, Graph):
            W = W.transform("R")
            Wyf = W.lag(y_filtered_)
        else:
            W.transform = "r"
            Wyf = lag_spatial(W, y_filtered_) # TODO: graph
        self.connectivity = W
        self.partials_ = np.column_stack((y_filtered_, Wyf))
        y_out = self.y_filtered_
        self.association_ = ((y_out * Wyf) / (y_out.T @ y_out) * (W.n - 1)).flatten()
        if self.permutations > 0:
            self._crand()
            self.significance_ = calculate_significance(self.association_, self.reference_distribution_, alternative=self.alternative)
        quads = np.array([[3,2,4,1]]).reshape(2,2)
        left_component_cluster = (y_filtered_ > 0).astype(int)
        right_component_cluster = (Wyf > 0).astype(int)
        quads = quads[left_component_cluster, right_component_cluster]
        self.labels_ = quads.squeeze()
        return self

    def _part_regress_transform(self, y, X):
        """If the object has a _transformer, use it; otherwise, fit it."""
        if hasattr(self, "_transformer"):
            ypart = y - self._transformer(X)
        else:
            from sklearn.linear_model import LinearRegression

            self._transformer = LinearRegression().fit(X, y).predict
            ypart = self._part_regress_transform(y, X)
        return ypart

    def _crand(self):
        """Cribbed from esda.Moran_Local
        conditional randomization
        for observation i with ni neighbors,  the candidate set cannot include
        i (we don't want i being a neighbor of i). we have to sample without
        replacement from a set of ids that doesn't include i. numpy doesn't
        directly support sampling wo replacement and it is expensive to
        implement this. instead we omit i from the original ids,  permute the
        ids and take the first ni elements of the permuted ids as the
        neighbors to i in each randomization.
        """
        _, z = self.partials_.T
        lisas = np.zeros((self.connectivity.n, self.permutations))
        n_1 = self.connectivity.n - 1
        prange = list(range(self.permutations))
        if isinstance(self.connectivity, Graph):
            k = self.connectivity.cardinalities.max() + 1
        else:
            k = self.connectivity.max_neighbors + 1
        nn = self.connectivity.n - 1
        rids = np.array([np.random.permutation(nn)[0:k] for i in prange])
        ids = np.arange(self.connectivity.n)
        if hasattr(self.connectivity, "id_order"):
            ido = self.connectivity.id_order
        else:
            ido = self.connectivity.unique_ids.values
        w = [self.connectivity.weights[ido[i]] for i in ids]
        wc = [self.connectivity.cardinalities[ido[i]] for i in ids]

        for i in tqdm(range(self.connectivity.n), desc="Simulating by site"):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            tmp = z[idsi[rids[:, 0 : wc[i]]]]
            lisas[i] = z[i] * (w[i] * tmp).sum(1)
        self.reference_distribution_ = (n_1 / (z * z).sum()) * lisas


MoranLocalConditional.__init__.__doc__ = MoranLocalPartial.__init__.__doc__.replace(
    "Partial", "Conditional"
)
