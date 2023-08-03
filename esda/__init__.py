"""
:mod:`esda` --- Exploratory Spatial Data Analysis
=================================================

"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import adbscan, shape  # noqa F401
from .gamma import Gamma  # noqa F401
from .geary import Geary  # noqa F401
from .geary_local import Geary_Local  # noqa F401
from .geary_local_mv import Geary_Local_MV  # noqa F401
from .getisord import G, G_Local  # noqa F401
from .join_counts import Join_Counts  # noqa F401
from .join_counts_local import Join_Counts_Local  # noqa F401
from .join_counts_local_bv import Join_Counts_Local_BV  # noqa F401
from .join_counts_local_mv import Join_Counts_Local_MV  # noqa F401
from .lee import Spatial_Pearson, Spatial_Pearson_Local  # noqa F401
from .losh import LOSH  # noqa F401
from .map_comparison import (
    areal_entropy,
    completeness,  # noqa F401
    external_entropy,
    homogeneity,
    overlay_entropy,
)
from .moran import (
    Moran,
    Moran_BV,
    Moran_BV_matrix,
    Moran_Local,  # noqa F401
    Moran_Local_BV,
    Moran_Local_Rate,
    Moran_Rate,
)
from .silhouettes import boundary_silhouette, path_silhouette  # noqa F401
from .smaup import Smaup  # noqa F401
from .topo import isolation, prominence  # noqa F401
from .util import fdr  # noqa F401

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("esda")
