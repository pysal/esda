# Spatial Pattern Diagnostics

**Question:** *How does spatial dependence change as the neighbourhood expands?*

A single global statistic collapses all scales into one number. The correlogram reveals the *scale structure* of spatial dependence by computing an autocorrelation statistic at a series of increasing distances or neighbourhood sizes.

**Key concepts:**

- **Distance-band correlogram** — computes the statistic for all pairs of observations within each successive distance band. Reveals whether autocorrelation drops off smoothly, has a secondary peak, or oscillates.
- **KNN correlogram** — steps through $k = 1, 2, \ldots$ nearest neighbours. Useful when observations are unevenly spaced and fixed distance bands would produce empty or overloaded bands.
- **Nonparametric (LOWESS)** — fits a locally weighted regression of pairwise products against pairwise distances, providing a smooth curve without pre-defined bins.

The correlogram accepts any `esda` global statistic (Moran's $I$, Geary's $C$, Getis-Ord $G$), so the profile shape changes depending on which aspect of spatial association is being measured.

```{toctree}
:maxdepth: 1

correlogram
```
