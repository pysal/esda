# Local Spatial Structure

**Question:** *Where are clusters or spatial outliers located?*

Local methods assign each observation its own statistic, identifying which specific units are driving the global pattern — or where the global pattern breaks down.

**Choosing a method:**

- **Local Moran's I (LISA)** — decomposes Moran's $I$ into per-unit contributions. Produces four quadrant labels: High-High (hot spot), Low-Low (cold spot), High-Low and Low-High (spatial outliers).
- **Local Geary** — analogous to global Geary's C at the unit level. Distinguishes *clusters* (similar neighbours) from *outliers* (dissimilar neighbours) without using the global mean as a reference.
- **$G_i$ (Getis-Ord local)** — identifies hotspots and coldspots based on raw value concentration; does not detect spatial outliers. Preferred when magnitude rather than relative position matters.
- **LOSH** — measures *local variance* rather than the local mean. Most useful in combination with a mean statistic to detect transitional or boundary zones.
- **Local Join Counts** — the local version of join count analysis for binary variables; detects co-location of positive events.
- **Multivariate Moran** — extends local Moran to relationships between two or more variables.

All local statistics carry multiple-comparison risk; applying a false-discovery-rate correction (e.g., `esda.fdr`) before mapping is recommended.

```{toctree}
:maxdepth: 1

local_spatial_autocorrelation
local_geary
local_getisord
localjoincounts
LOSH
multivariable_moran
```
