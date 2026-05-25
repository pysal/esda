import numpy as np

from .getisord import G_Local
from .losh import LOSH
from .moran import Moran_Local


class LocalCrossPlot:
    """Combine local statistics into a G-I-LOSH cross plot.

    The local G-I-LOSH cross plot is a joint diagnostic that places standardized
    Getis-Ord :math:`G_i` values on the x-axis and standardized Local
    Moran statistics on the y-axis while scaling symbol sizes by local
    spatial heteroscedasticity (LOSH), proposed by :cite:`westerholt_2026_19421814`.
    This provides a compact view of
    local clustering, local association, and local variance structure in
    a single graphic.

    For details refer to :cite:`westerholt_2026_19421814`.

    Parameters
    ----------
    connectivity : W | Graph, optional
        Spatial weights object aligned with the observed values.
    permuations : int, default=999
        Number of random permutations used when fitting
        :class:`~esda.moran.Moran_Local` and
        :class:`~esda.getisord.G_Local`.
    star : bool, default=False
        Whether to include the focal observation in the Getis-Ord local
        statistic.
    n_jobs : int, default=-1
        Number of parallel workers for permutation-based inference in the
        local Moran and local Getis-Ord statistics.
    seed : int, optional
        Random seed forwarded to permutation-based local statistics.
    island_weight : float, default=0
        Weight assigned to the synthetic neighbor used for islands in the
        local Moran and local Getis-Ord calculations.
    inference : str, optional
        Inference method for :class:`~esda.losh.LOSH`. See
        :class:`~esda.losh.LOSH` for supported options.
    a : int or float, default=2
        Residual exponent passed to :meth:`esda.losh.LOSH.fit`. The
        default corresponds to a variance-based LOSH measure.

    Attributes
    ----------
    connectivity : W | Graph or None
        Spatial weights object used to fit the component estimators.
    permutations : int
        Number of permutations used for local Moran and local Getis-Ord
        inference.
    losh_ : LOSH
        Fitted LOSH estimator.
    moran_local_ : Moran_Local
        Fitted local Moran estimator.
    g_local_ : G_Local
        Fitted local Getis-Ord estimator.
    """

    def __init__(
        self,
        connectivity=None,
        permuations=999,
        star=False,
        n_jobs=-1,
        seed=None,
        island_weight=0,
        inference=None,
        a=2,
    ):
        self.connectivity = connectivity
        self.permutations = permuations
        self.star = star
        self.n_jobs = n_jobs
        self.seed = seed
        self.island_weight = island_weight
        self.inference = inference
        self.a = a

    def fit(self, y):
        """Fit the component local statistics used in the plot.

        Parameters
        ----------
        y : array_like
            One-dimensional array of observed values aligned with
            ``connectivity``.

        Returns
        -------
        LocalCrossPlot

        Notes
        -----
        Fitting computes and stores:

        - :class:`~esda.losh.LOSH` for local spatial heteroscedasticity,
        - :class:`~esda.moran.Moran_Local` for local spatial association,
        - :class:`~esda.getisord.G_Local` for local concentration.
        """
        self.y = y

        self.losh_ = LOSH(self.connectivity, inference=self.inference).fit(y, a=self.a)
        self.moran_local_ = Moran_Local(
            y,
            self.connectivity,
            permutations=self.permutations,
            n_jobs=self.n_jobs,
            seed=self.seed,
            island_weight=self.island_weight,
        )
        self.g_local_ = G_Local(
            y,
            self.connectivity,
            star=self.star,
            permutations=self.permutations,
            n_jobs=self.n_jobs,
            seed=self.seed,
            island_weight=self.island_weight,
        )

        return self

    def plot(
        self,
        crit_value=0.05,
        losh_scaling_factor=10,
        linewidth=0.5,
        ax=None,
        legend=False,
    ):
        """Draw the local cross plot.

        Parameters
        ----------
        crit_value : float, default=0.05
            The critical value for significance.
        losh_scaling_factor : float, default=10
            Multiplicative factor applied to ``exp(losh_.Hi)`` when
            converting LOSH values into marker areas.
        linewidth : float, default=0.5
            Line width for marker outlines.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the plot. If omitted, a new figure and
            axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.

        Notes
        -----
        The plot uses the following encodings:

        - x-axis: standardized local Getis-Ord :math:`G_i^*`,
        - y-axis: permutation-standardized Local Moran statistic,
        - marker size: ``exp(LOSH)``,
        - marker color: significance/sign combinations of local
          Getis-Ord and Local Moran results.
        """
        import matplotlib.pyplot as plt

        g = self.g_local_.Zs
        i = self.moran_local_.z_sim

        moran_sig = self.moran_local_.p_sim < crit_value
        g_sig = self.g_local_.p_sim < crit_value

        color = np.array(["lightgrey"] * len(g), dtype="O")
        color[(g < 0) & g_sig] = "darkblue"
        color[(g < 0) & g_sig & moran_sig] = "royalblue"
        color[(g < 0) & ~g_sig & moran_sig] = "dodgerblue"
        color[(g > 0) & g_sig] = "maroon"
        color[(g > 0) & g_sig & moran_sig] = "firebrick"
        color[(g > 0) & ~g_sig & moran_sig] = "coral"

        sig_mask = color != "lightgrey"

        if not ax:
            _, ax = plt.subplots()

        # significant on top
        sc = ax.scatter(
            x=g[sig_mask],
            y=i[sig_mask],
            s=np.exp(self.losh_.Hi[sig_mask]) * losh_scaling_factor,
            facecolor="none",
            edgecolor=color[sig_mask],
            linewidth=linewidth,
            zorder=1,
        )

        # insignificant under
        ax.scatter(
            x=g[~sig_mask],
            y=i[~sig_mask],
            s=np.exp(self.losh_.Hi[~sig_mask]) * losh_scaling_factor,
            facecolor="none",
            edgecolor=color[~sig_mask],
            linewidth=linewidth,
            zorder=0,
        )

        ax.set_xlabel(f"Getis-Ord $G{'^*' if self.g_local_.star else ''}_i$")
        ax.set_ylabel("Moran's $I_i$")
        ax.axvline(0, color="silver", linestyle="dashed")
        ax.axhline(0, color="silver", linestyle="dashed")

        if legend:
            handles, labels = sc.legend_elements(prop="sizes", num=4)
            for h in handles:
                h.set_fillstyle("none")
            labels = [(np.log(float(lab[14:-2])) / 10).round(2) for lab in labels]
            ax.legend(handles, labels, title="LOSH")

        return ax

    @classmethod
    def from_estimators(cls, g_local, moran_local, losh):
        """Construct a plotter from pre-fitted component estimators.

        Parameters
        ----------
        g_local : G_Local
            Fitted local Getis-Ord estimator.
        moran_local : Moran_Local
            Fitted local Moran estimator.
        losh : LOSH
            Fitted LOSH estimator.

        Returns
        -------
        LocalCrossPlot
            Plotter populated with the provided estimators.

        Notes
        -----
        This constructor is useful when the component estimators have
        already been fit elsewhere or when custom settings were used for
        each statistic independently.
        """
        wp = cls()

        if isinstance(moran_local.z_sim, float):
            raise ValueError(
                "Moran_Local needs to be fitted with keep_simulations=True."
            )

        wp.losh_ = losh
        wp.moran_local_ = moran_local
        wp.g_local_ = g_local

        return wp
