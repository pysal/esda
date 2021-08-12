from .moran import (
    Moran,
    Moran_Local,
    Moran_BV,
    Moran_Local_BV,
    Moran_Rate,
    Moran_Local_Rate,
)
from .geary import Geary
from .gamma import Gamma
from .geary_local import Geary_Local
from .geary_local_mv import Geary_Local_MV
from .getisord import G, G_Local
from .join_counts import Join_Counts
from .join_counts_local import Join_Counts_Local
from .join_counts_local_bv import Join_Counts_Local_BV
from .join_counts_local_mv import Join_Counts_Local_MV

# from .lee import Spatial_Pearson # no solution yet for sklearn style classes
# from .losh import LOSH
import inspect

for klass in (
    Moran,
    Moran_Local,
    Moran_BV,
    Moran_Local_BV,
    Moran_Rate,
    Moran_Local_Rate,
    Geary,
    Gamma,
    Geary_Local,
    Geary_Local_MV,
    G,
    G_Local,
    Join_Counts,
    Join_Counts_Local,
    Join_Counts_Local_BV,
    Join_Counts_Local_MV,
):
    assert hasattr(klass, "_statistic"), f"{klass} has no _statistic"
    assert not callable(klass._statistic), f"{klass}._statistic is callable"
    klassname = klass.__name__
    name = klass.__name__.lower()
    if klassname == "LOSH":
        defn = f"def {name}(*args, **kwargs):\n\tobj = {klassname}(*args, **kwargs)\n\treturn obj._statistic, obj.pval"
    elif klassname == "Spatial_Pearson":
        defn = f"def {name}(*args, **kwargs):\n\tobj = {klassname}(*args, **kwargs)\n\treturn obj._statistic, obj.significance_"
    else:
        defn = f"def {name}(*args, **kwargs):\n\tobj = {klassname}(*args, **kwargs)\n\treturn obj._statistic, obj.p_sim"
    exec(defn)
    exec(f"{name}.__doc__ = {klassname}.__doc__")
    init_sig = inspect.signature(klass)
    globals()[name].__signature__ = init_sig
    del globals()[klassname]

for klass in (LOSH, Spatial_Pearson):
    # sklearn style...
    pass

del klassname
del klass
del name
del init_sig
del defn
del inspect
