# User Guide

`esda` covers a range of exploratory spatial data analysis methods.
To help get you oriented, start with the **question you are trying to answer**, then follow the relevant section.

New to `esda`? Start with the [Getting Started](getting_started) overview, which walks through the key concepts — spatial weights, spatial lag, autocorrelation statistics, and permutation inference — using a single dataset from beginning to end.

## Guiding Questions

- **Is there spatial autocorrelation?** → [Global Spatial Autocorrelation](global)
  Methods: Moran's I, Geary's C, Getis-Ord G, Join Counts

- **Where are clusters or outliers?** → [Local Spatial Autocorrelation](local)
  Methods: Local Moran's I (LISA), Local Geary, G_i*, LOSH, Local Join Counts, Multivariate Moran

- **How does spatial dependence change across distance?** → [Spatial Pattern Diagnostics](diagnostics)
  Methods: Spatial correlogram (distance bands, KNN, nonparametric)

- **Where do point clusters form?** → [Spatial Clustering](clustering)
  Methods: A-DBSCAN

- **What are the shapes or geometry of features?** → [Shape and Geometry Analysis](geometry)
  Methods: Shape compactness measures, geo-silhouettes

- **How are units connected structurally?** → [Topology](topology)
  Methods: Isolation, Prominence

```{toctree}
:hidden:
:maxdepth: 1

getting_started
```
