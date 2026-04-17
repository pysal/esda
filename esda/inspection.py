import numpy as np

from .getisord import G_Local
from .losh import LOSH
from .moran import Moran_Local


class WesterholtPlot:
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
            permutations=self.permutations,
            n_jobs=self.n_jobs,
            seed=self.seed,
            island_weight=self.island_weight,
        )

        return self

    def plot(self, losh_scaling_factor=10, linewidth=0.5, ax=None):
        import matplotlib.pyplot as plt

        g = self.g_local_.Zs
        i = self.moran_local_.z_sim

        moran_sig = self.moran_local_.p_sim < 0.05
        g_sig = self.g_local_.p_sim < 0.05

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
        ax.scatter(
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

        ax.set_xlabel("Getis-Ord $G^*_i$")
        ax.set_ylabel("Moran's $I$")
        ax.axvline(0, color="silver", linestyle="dashed")
        ax.axhline(0, color="silver", linestyle="dashed")

        return ax

    @classmethod
    def from_estimators(cls, g_local, moran_local, losh):
        wp = cls()

        wp.losh_ = losh
        wp.moran_local_ = moran_local
        wp.g_local_ = g_local

        return wp
