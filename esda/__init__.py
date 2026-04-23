"""
:mod:`esda` --- Exploratory Spatial Data Analysis
=================================================

"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import adbscan, shape
from .correlogram import correlogram
from .gamma import Gamma
from .geary import Geary
from .geary_local import Geary_Local
from .geary_local_mv import Geary_Local_MV
from .getisord import G, G_Local
from .inspection import LocalCrossPlot
from .join_counts import Join_Counts
from .join_counts_local import Join_Counts_Local
from .join_counts_local_bv import Join_Counts_Local_BV
from .join_counts_local_mv import Join_Counts_Local_MV
from .lee import Spatial_Pearson, Spatial_Pearson_Local
from .losh import LOSH
from .map_comparison import (
    areal_entropy,
    completeness,
    external_entropy,
    homogeneity,
    overlay_entropy,
)
from .moran import (
    Moran,
    Moran_BV,
    Moran_BV_matrix,
    Moran_Local,
    Moran_Local_BV,
    Moran_Local_Rate,
    Moran_Rate,
    plot_moran_facet,
)
from .moran_local_mv import MoranLocalConditional, MoranLocalPartial
from .silhouettes import boundary_silhouette, path_silhouette
from .smaup import Smaup
from .topo import isolation, prominence
from .util import fdr

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("esda")
