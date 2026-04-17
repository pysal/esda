import numpy as np

from .getisord import G_Local
from .losh import LOSH
from .moran import Moran_Local


class WesterholtPlot:
    def __init__(self, connectivity=None):
        self.connectivity = connectivity

    def fit(self, y):
        self.y = y

        self.losh_ = LOSH(self.connectivity).fit(y)
        self.moran_local_ = Moran_Local(y, self.connectivity)
        self.g_local_ = G_Local(y, self.connectivity)

        return self

    def plot(self):
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

        f, ax = plt.subplots()

        # significant on top
        ax.scatter(
            x=g[sig_mask],
            y=i[sig_mask],
            s=np.exp(self.losh_.Hi[sig_mask]) * 10,
            facecolor="none",
            edgecolor=color[sig_mask],
            linewidth=0.5,
            zorder=1,
        )

        # insignificant under
        ax.scatter(
            x=g[~sig_mask],
            y=i[~sig_mask],
            s=np.exp(self.losh_.Hi[~sig_mask]) * 10,
            facecolor="none",
            edgecolor=color[~sig_mask],
            linewidth=0.5,
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
