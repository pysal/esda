import numpy as np
import esda
from libpysal.weights import lag_spatial

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


class Partial_Moran_Local(object):
    def __init__(
        self, y, X, W, permutations=999, dedupe=True, unit_scale=True, mvquads=True
    ):
        """
        Compute the Multivariable Local Moran statistics under partial dependence, as defined by :cite:`wolf2024confounded`

        Arguments
        ---------
        y               : (N,1) array
                          array of data that is the targeted "outcome" covariate
                          to compute the multivariable Moran's I
        X               : (N,3) array
                          array of data that is used as "confounding factors"
                          to account for their covariance with Y.
        W               : (N,N) weights object
                          a PySAL weights object. Immediately row-standardized.
        permutations    : int
                          the number of permutations to run for the inference,
                          driven by conditional randomization.


        Attributes
        ----------
        W      : The weights matrix inputted, but row standardized
        D      : The "design" matrix used in computation. If X is
                      not None, this will be [1 y X]
        R      : the "response" matrix used in computation. Will
                      always be the same shape as D and contain [1, Wy, Wy, ....]
        DtDi   : empirical parameter covariance matrix
                      the P x P matrix describing the variance and covariance
                      of y and X.
        P      : the number of parameters. 1 if X is not provided.
        lmos_   : the N,P matrix of multivariable LISA statistics.
                      the first column, lmos[:,1] is the LISAs corresponding
                      to the relationship between Wy and y conditioning on X.
        rlmos_  : the (N, permutations, P+1) realizations from the conditional
                  randomization to generate reference distributions for
                  each Local Moran statistic. rlmos_[:,:,1] pertain to
                  the reference distribution of y and Wy.
        quads_  : the (N, P) matrix of quadrant classifications for the
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
        """
        self._mvquads = mvquads
        y = np.asarray(y).reshape(-1, 1)
        W.transform = "r"
        y -= y.mean()
        if unit_scale:
            y /= y.std()
        X -= X.mean(axis=0)
        if unit_scale:
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
        self._left_component_ = (self.D @ self.DtDi) * (W.n - 1)
        self._lmos_ = self._left_component_ * self.R
        self.W = W
        self.permutations = permutations
        if permutations is not None:  # NOQA necessary to avoid None > 0
            if permutations > 0:
                if dedupe:
                    self._crand(y, X, W)
                else:
                    self._dupe_crand(y, X, W)

                self._rlmos_ *= W.n - 1
                self._p_sim_ = np.zeros((W.n, self.P + 1))
                for permutation in range(self.permutations):
                    self._p_sim_ += (
                        self._rlmos_[:, permutation, :] < self._lmos_
                    ).astype(int)
                self._p_sim_ /= self.permutations
                self._p_sim_ = np.minimum(self._p_sim_, 1 - self._p_sim_)

        component_quads = []
        for i, left in enumerate(self._left_component_.T):
            right = self.R[:, i]
            quads = (left < left.mean()).astype(int)
            quads += (right < right.mean()).astype(int) * 2 + 1
            quads[quads == 4] = 5
            quads[quads == 3] = 4
            quads[quads == 5] = 3
            component_quads.append(quads)
        self._partials_ = np.asarray(
            [
                np.vstack((left, right)).T
                for left, right in zip(self._left_component_.T, self.R.T)
            ]
        )

        uvquads = []
        for i, x_ in enumerate(self.D.T):
            if i == 0:
                continue
            quads = (self.R[:, 1].flatten() < self.R[:, 1].mean()).astype(int)
            quads += (x_.flatten() < x_.mean()).astype(int) * 2
            quads += 1
            quads[quads == 4] = 5
            quads[quads == 3] = 4
            quads[quads == 5] = 3
            uvquads.append(quads.flatten())

        self._uvquads_ = np.row_stack(uvquads).T
        self._mvquads_ = np.row_stack(component_quads).T
        if self._mvquads:
            self.quads_ = self._mvquads_[:, 1]
        else:
            self.quads_ = self._uvquads_[:, 1]

    def _make_data(self, z, X, W):
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
        max_neighbs = W.max_neighbors + 1
        pre_permutations = np.array(
            [np.random.permutation(N - 1)[0:max_neighbs] for i in prange]
        )
        straight_ids = np.arange(N)
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
            # shuffled_X = X[randomized_permutations, :] #these are still N-permutations, N-neighbs, N-covariates
            if X is None:
                local_data = np.array((1, y[i])).reshape(1, -1)
                shuffled_D = np.tile(
                    local_data, N_permutations
                ).T  # now N permutations by P
            else:
                local_data = np.array((1, y[i], *X[i])).reshape(-1, 1)
                shuffled_D = np.tile(
                    local_data, N_permutations
                ).T  # now N permutations by P
            shuffled_R = np.tile(shuffled_Wyi, self.P + 1)
            lmos[i] = (shuffled_R * shuffled_D) @ DtDi
        self._rlmos_ = lmos  # nobs, nperm, nvars

    def _dupe_crand(self, y, X, W):
        """
        This does a full resampling, allowing the self neighbor
        """
        N = W.n
        N_permutations = self.permutations
        DtDi = self.DtDi
        permutations = np.empty((N, N_permutations, self.P + 1))
        from collections import defaultdict

        n_dupes = defaultdict(int)
        W = self.W
        shuffs = [np.random.permutation(N) for _ in range(N_permutations)]
        for permutation in tqdm(range(N_permutations), desc="Simulating"):
            for observation in range(N):
                # copy the data
                yrand = y.copy()
                Xrand = X.copy()
                # make a shuffler
                shuff = shuffs[permutation]
                np.random.shuffle(shuff)
                # shufle the input y and X, but keep records aligned
                yrand = yrand[shuff, :]
                Xrand = Xrand[shuff, :]
                neighbors = shuff[W.neighbors[observation]]
                n_dupes[observation] += int(observation in neighbors)
                # reset the focal observation to its original value.
                yrand[observation] = y[observation]
                Xrand[observation] = X[observation]
                # make Wy, D, and R
                Drand, Rrand = self._make_data(yrand, Xrand, W)
                # use these in the p-values
                permutations[observation, permutation, :] = ((Drand * Rrand) @ DtDi)[
                    observation
                ]
        self._rlmos_ = np.asarray(permutations)

    @property
    def associations_(self):
        """
        The association between y and the local average of y,
        removing the correlation due to x and the local average of y
        """
        return self._lmos_[:, 1]

    @property
    def significances_(self):
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
        Simulated distribution of associations_, assuming that there is
          - no structural relationship between y and its local average;
          - the same observed structural relationship between y and x.
        """
        return self._rlmos_[:, :, 1]

    @property
    def labels_(self):
        """
        The classifications (in terms of cluster-type and outlier-type)
        for the associations_ statistics. If the quads requested are
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
        if self._mvquads:
            return self._mvquads_[:, 1]
        else:
            return self._uvquads_[:, 1]


class Auxiliary_Moran_Local(esda.Moran_Local):
    """
    Fit a local moran statistic for y after regressing out the
    effects of confounding X on y. A "stronger" version of the
    Partial_Moran statistic, as defined by :cite:`wolf2024confounded`
    """

    def __init__(
        self,
        y,
        X,
        W,
        permutations=999,
        unit_scale=True,
        transformer=None,
    ):
        """
        Fit a local Moran statistic on the regression residuals

        Arguments
        ---------
        y               : (N,1) array
                          array of data that is the targeted "outcome" covariate
                          to compute the multivariable Moran's I
        X               : (N,3) array
                          array of data that is used as "confounding factors"
                          to account for their covariance with Y.
        W               : (N,N) weights object
                          a PySAL weights object. Immediately row-standardized.
        permutations    : int (default: 999)
                          the number of permutations to run for the inference,
                          driven by conditional randomization.
        unit_scale      : bool (default: True)
                          whether or not to convert the input data to a unit normal scale.
                          data is ALWAYS centered, but variance will remain unadjusted.
        transformer     : callable (default: scikit regression)
                          should transform X into a predicted y. If not provided, will use
                          the standard scikit OLS regression of y on X.
        """
        y -= y.mean()
        X -= X.mean(axis=0)
        if unit_scale:
            y /= y.std()
            X /= X.std(axis=0)
        self.y = y
        self.X = X
        W.transform = "r"
        self.W = W
        y_filtered_ = self._part_regress_transform(y, X)
        Wyf = lag_spatial(self.W, y_filtered_)
        self.partials_ = np.column_stack((Wyf, y_filtered_))
        self.permutations = permutations
        y_out = self.y_filtered_
        self.associations_ = ((y_out * Wyf) / (y_out.T @ y_out) * (W.n - 1)).flatten()
        self._crand()
        p_sim = (self.reference_distribution_ < self.associations_[:, None]).mean(
            axis=1
        )
        self.significances_ = np.minimum(p_sim, 1 - p_sim)
        quads = (y_out.flatten() < y_out.mean()).astype(int)
        quads += (Wyf.flatten() < Wyf.mean()).astype(int) * 2
        quads += 1
        quads[quads == 3] = 5
        quads[quads == 4] = 3
        quads[quads == 5] = 4
        self.labels_ = quads

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
        _, z = self.partials.T
        lisas = np.zeros((self.W.n, self.permutations))
        n_1 = self.W.n - 1
        prange = list(range(self.permutations))
        k = self.W.max_neighbors + 1
        nn = self.W.n - 1
        rids = np.array([np.random.permutation(nn)[0:k] for i in prange])
        ids = np.arange(self.W.n)
        ido = self.W.id_order
        w = [self.W.weights[ido[i]] for i in ids]
        wc = [self.W.cardinalities[ido[i]] for i in ids]

        for i in tqdm(range(self.W.n), desc="Simulating by site"):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            tmp = z[idsi[rids[:, 0 : wc[i]]]]
            lisas[i] = z[i] * (w[i] * tmp).sum(1)
        self.reference_distribution_ = (n_1 / (z * z).sum()) * lisas


Auxiliary_Moran_Local.__init__.__doc__ = Partial_Moran_Local.__init__.__doc__.replace(
    "Partial", "Auxiliary"
)
