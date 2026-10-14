
# Spatial Clustering
**Question:** *Where do point clusters form, and how fuzzy are their boundaries?*

Density-based spatial clustering groups points into clusters based on proximity and density without requiring you to specify the number of clusters in advance.

**A-DBSCAN (Adaptive DBSCAN):**

A-DBSCAN extends the classic DBSCAN algorithm by running many bootstrap replicates of the clustering and measuring how consistently each point is assigned to the same cluster. The result is a *vote share* for each point — how often it was labelled the same way across replicates — which quantifies boundary uncertainty. Points with low vote shares lie in ambiguous transition zones; points with high vote shares are in cluster cores.


```{toctree}
:maxdepth: 1

adbscan_berlin_example
```
