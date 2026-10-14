# Shape and Geometry Analysis

**Question:** *What are the geometric or morphological properties of features or clusters?*

These methods characterise the shape and spatial configuration of polygons independently of any attribute values.

**Choosing a method:**

- **Shape measures** (`esda.shape`) — a suite of compactness and elongation indices (isoperimetric quotient, convex hull ratio, moment of inertia, and more). Useful for comparing the regularity of electoral districts, census units, or natural features.
- **Geo-silhouettes** (`boundary_silhouette`, `path_silhouette`) — spatial analogues of the silhouette score from cluster analysis. Measure how well a cluster assignment is supported by the underlying geography: a high score means observations are closer to members of their own cluster than to members of neighbouring clusters.

Shape measures operate on individual polygons; geo-silhouettes require a cluster label and a spatial graph or distance matrix.

```{toctree}
:maxdepth: 1

shape-measures
geosilhouettes
```
