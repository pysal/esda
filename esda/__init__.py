"""
:mod:`esda` --- Exploratory Spatial Data Analysis
=================================================

"""
from . import adbscan
from .gamma import Gamma
from .geary import Geary
from .geary_local import Geary_Local
from .geary_local_mv import Geary_Local_MV
from .getisord import G, G_Local
from .join_counts import Join_Counts
from .join_counts_local import Join_Counts_Local
from .join_counts_local_bv import Join_Counts_Local_BV
from .join_counts_local_mv import Join_Counts_Local_MV
from .lee import Spatial_Pearson, Spatial_Pearson_Local
from .losh import LOSH
from .map_comparison import (
    external_entropy,
    completeness,
    homogeneity,
    overlay_entropy,
    areal_entropy,
)
from .moran import (
    Moran,
    Moran_BV,
    Moran_BV_matrix,
    Moran_Local,
    Moran_Local_BV,
    Moran_Rate,
    Moran_Local_Rate,
)
from .silhouettes import path_silhouette, boundary_silhouette
from . import shape
from .smaup import Smaup
from .topo import prominence, isolation
from .util import fdr

from . import _version

__version__ = _version.get_versions()["version"]
