# Exploratory Spatial Data Analysis in PySAL

![tag](https://img.shields.io/github/v/release/pysal/esda?include_prereleases&sort=semver)
[![Continuous Integration](https://github.com/pysal/esda/actions/workflows/testing.yml/badge.svg)](https://github.com/pysal/esda/actions/workflows/testing.yml) 
[![codecov](https://codecov.io/gh/pysal/esda/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/esda)
[![DOI](https://zenodo.org/badge/81873636.svg)](https://zenodo.org/badge/latestdoi/81873636)

Methods for testing for global and local autocorrelation in areal unit data.

## Documentation
- [Home](https://pysal.org/esda)
- [Tutorial](https://pysal.org/esda/tutorial.html)
- [API](https://pysal.org/esda/api.html)

## Installation

Install `esda` by running:

### conda-forge

*preferred*

```
$ conda install -c conda-forge esda
```

### PyPI

```
$ pip install esda
```

### GitHub

```
$ pip install git+https://github.com/pysal/esda@main
```

## Requirements

*see currently supported versions in [`pyproject.toml[dependencies]`](https://github.com/pysal/esda/blob/main/pyproject.toml)*

- `geopandas`
- `libpysal`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `shapely`

## Optional dependencies

*see currently supported versions in [`pyproject.toml[project.optional-dependencies]`](https://github.com/pysal/esda/blob/main/pyproject.toml)*

- `matplotlib` - required for `esda.moran.explore()`
- `numba` - used to accelerate computational geometry and permutation-based statistical inference.
- `rtree` - required for `esda.topo.isolation()`

## Contribute

PySAL-esda is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/esda/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](https://github.com/pysal/pysal/wiki) and open a [pull request](https://github.com/pysal/esda). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/esda/graphs/contributors).

## Support

If you are having issues, please talk to us in the [`esda` Discord channel](https://discord.gg/Re46DjyB9U).

## License

The project is licensed under the [BSD 3-Clause license](https://github.com/pysal/esda/blob/main/LICENSE).

## Funding

[<img align="middle" src="https://github.com/pysal/esda/blob/main/docs/_static/images/nsf_logo.jpg" width="100">](https://www.nsf.gov/index.jsp) National Science Foundation Award #1421935: [New Approaches to Spatial Distribution Dynamics](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1421935)
