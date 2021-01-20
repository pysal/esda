# Exploratory Spatial Data Analysis in PySAL
[![unittests](https://github.com/pysal/esda/workflows/.github/workflows/unittests.yml/badge.svg)](https://github.com/pysal/esda/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml)
[![codecov](https://codecov.io/gh/pysal/esda/branch/master/graph/badge.svg)](https://codecov.io/gh/pysal/esda)
[![DOI](https://zenodo.org/badge/81873636.svg)](https://zenodo.org/badge/latestdoi/81873636)

Methods for  testing for global and local autocorrelation in areal unit data.

## Documentation
- [Home](https://pysal.org/esda)
- [Tutorial](https://pysal.org/esda/tutorial.html)
- [API](https://pysal.org/esda/api.html)


## Installation

Install esda by running:

```
$ pip install esda
```


## Requirements

- libpysal

### Optional dependencies

- `numba`, version `0.50.1` or greater, is used to accelerate computational geometry and permutation-based statistical inference. Unfortunately, versions before `0.50.1` may cause some local statistical functions to break, so please ensure you have `numba>=0.50.1` installed. 

## Contribute

PySAL-esda is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/esda/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](https://github.com/pysal/pysal/wiki) and open a [pull request](https://github.com/pysal/esda). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/esda/graphs/contributors).


## Support

If you are having issues, please talk to us in the [gitter room](https://gitter.im/pysal/pysal).
 
 
## License

The project is licensed under the [BSD 3-Clause license](https://github.com/pysal/esda/blob/master/LICENSE).


## Funding

[<img align="middle" src="figs/nsf_logo.jpg" width="100">](https://www.nsf.gov/index.jsp) National Science Foundation Award #1421935: [New Approaches to Spatial Distribution Dynamics](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1421935)
