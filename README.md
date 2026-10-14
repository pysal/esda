# esda

![tag](https://img.shields.io/github/v/release/pysal/esda?include_prereleases&sort=semver)
[![Continuous Integration](https://github.com/pysal/esda/actions/workflows/testing.yml/badge.svg)](https://github.com/pysal/esda/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/pysal/esda/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/esda)
[![DOI](https://zenodo.org/badge/81873636.svg)](https://zenodo.org/badge/latestdoi/81873636)

## Introduction

**esda** is a library for exploratory spatial data analysis (ESDA). It is part of the PySAL (Python Spatial Analysis Library) ecosystem and provides methods for measuring, testing, and visualizing spatial autocorrelation and spatial association patterns in geospatial data.

Built on top of NumPy, SciPy, GeoPandas, and libpysal, esda offers a comprehensive collection of global and local statistics for understanding spatial structure in areal and point-referenced data.

### What can esda do?

esda aims to provide a broad collection of methods for exploratory analysis of spatial data, enabling researchers and practitioners to identify spatial structure before proceeding to formal modeling.

Some of the functionality that esda offers:

* Compute global spatial autocorrelation statistics such as Moran’s I, Geary’s C, and Getis-Ord G.
* Compute local indicators of spatial association (LISA), including Local Moran, Local Geary, and Local Getis-Ord statistics.
* Analyze binary and categorical spatial patterns using join-count statistics.
* Explore multivariate spatial association with bivariate and multivariate Moran statistics.
* Measure spatial clustering, hot spots, and cold spots.
* Quantify shape regularity and geometric characteristics of spatial features.
* Support permutation-based inference for statistical significance testing.
* Integrate seamlessly with GeoPandas, libpysal, and the broader PySAL ecosystem.

See the [User Guide](http://pysal.org/esda/stable/user-guide/index.html) for more details.


## Installation

Install the latest release from PyPI:

```
pip install esda
```

Or install using conda-forge:

```
conda install -c conda-forge esda
```



## Relationship to PySAL

esda is one of the core packages in the PySAL ecosystem and works closely with:

* libpysal – spatial weights and data structures
* spreg – spatial econometric models
* segregation – segregation metrics
* pointpats – point pattern analysis
* giddy – spatial dynamics and mobility analysis

Learn more about the PySAL ecosystem at:

<https://pysal.org>

## Contributing


PySAL-esda is under active development and contributors are welcome.


Repository:

<https://github.com/pysal/esda>


If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/esda/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](https://github.com/pysal/pysal/wiki) and open a [pull request](https://github.com/pysal/esda). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/esda/graphs/contributors).

## Support

If you are having issues, please talk to us in the [`esda` Discord channel](https://discord.gg/Re46DjyB9U).


## Citation

If you use esda in research, please cite PySAL and the relevant methodological references associated with the statistics you employ:

```
@software{esda_2026,
  author       = {Sergio Rey and
                  Levi John Wolf and
                  James Gaboardi and
                  Dani Arribas-Bel and
                  Lee Hachadoorian and
                  Martin Fleischmann and
                  Wei Kang and
                  eli knaap and
                  mhwang4 and
                  Jay Laura and
                  Philip Stephens and
                  Charles Schmidt and
                  Stefanie Lumnitz and
                  David C. Folch and
                  Juan C Duque and
                  Luc Anselin and
                  Nicholas Malizia and
                  Filipe and
                  Thomas Louf and
                  Germano Barcelos and
                  Josiah Parry and
                  Michael Rariden and
                  matthewborish and
                  Jeff Sauer and
                  JasonSteelmanCoder and
                  Leo Morales and
                  mlyons-tcc and
                  Mridul Seth and
                  Nathaniel M. Beaver},
  title        = {pysal/esda: v2.9.0},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.9.0},
  doi          = {10.5281/zenodo.19140557},
  url          = {https://doi.org/10.5281/zenodo.19140557},
  swhid        = {swh:1:dir:52277964ad409a3710c0b71c80b787105b7f4028
                   ;origin=https://doi.org/10.5281/zenodo.1403275;vis
                   it=swh:1:snp:102da65f4352acd135660b6bc187cae834329
                   806;anchor=swh:1:rel:fb141a13a877f6c7cf991226c123f
                   1569edce2f1;path=pysal-esda-e0975f5
                  },
}
```

## License

The project is licensed under the [BSD 3-Clause license](https://github.com/pysal/esda/blob/main/LICENSE.txt).


## Funding

National Science Foundation Award #1421935: [New Approaches to Spatial Distribution Dynamics](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1421935)
