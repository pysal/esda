# Global Spatial Structure

**Question:** *Is there spatial autocorrelation across my entire study area?*

Global statistics return a single summary value for the whole dataset, answering whether the overall pattern is more clustered or dispersed than expected under spatial randomness.

**Choosing a method:**

- **Moran's I** — the standard choice for continuous variables. Compares each observation to its mean-centred value and its neighbours' mean-centred values; positive $I$ signals clustering, negative $I$ signals dispersion.
- **Geary's C** — also for continuous variables but based on *squared differences* between neighbours rather than cross-products. More sensitive to local contrasts; $C < 1$ is positive autocorrelation, $C > 1$ is negative.
- **Getis-Ord G** — tests for *spatial concentration* of raw (non-negative) values rather than deviations from the mean. Well-suited to counts, prices, and rates where the magnitude matters.
- **Join Counts** — designed for binary or categorical variables; counts how often similar or dissimilar categories share a boundary.

All methods support permutation-based inference, which avoids distributional assumptions and is recommended for small samples or skewed data.

```{toctree}
:maxdepth: 1

global_morans_i
geary
getisord
join_counts
```
