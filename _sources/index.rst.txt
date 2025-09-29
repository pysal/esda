.. documentation master file

ESDA: Exploratory Spatial Data Analysis
=======================================

ESDA is an open-source Python library for the exploratory analysis of spatial data. A subpackage of `PySAL`_ (Python Spatial Analysis Library), it is under active development and includes methods for global and local spatial autocorrelation analysis.

.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-12">
            <a href="https://nbviewer.org/github/pysal/esda/blob/main/notebooks/spatial_autocorrelation_for_areal_unit_data.ipynb" class="thumbnail">
                <img src="_static/images/prices.png" class="img-fluid center-block w-100">
                <div class="caption text-center">
                <h6>Spatial Autocorrelation for continuous and discrete data</h6>
                </div>
            </a>
        </div>
        </div>
        <div class="row equal-height">
        <div class="col-6">
            <a href="https://nbviewer.org/github/pysal/esda/blob/main/notebooks/shape-measures.ipynb" class="thumbnail">
                <img src="_static/images/missipq.png" class="img-fluid center-block w-100">
                <div class="caption text-center">
                <h6>Shape Regularity Statistics</h6>
                </div>
            </a>
        </div>
        <div class="col-6">
            <a href="https://nbviewer.org/github/pysal/esda/blob/main/notebooks/geosilhouettes.ipynb" class="thumbnail">
                <img src="_static/images/silhouettes.png" class="img-fluid center-block w-100">
                <div class="caption text-center">
                <h6>Boundary and Region Statistics
                </h6>
                </div>
            </a>
        </div>
        </div>
        <div class='row equal-height'>
        <div class="col-6">
        <a href="https://nbviewer.org/github/pysal/esda/blob/main/notebooks/multivariable_moran.ipynb" class="thumbnail">
                <img src="_static/images/mvmoran.png" class="img-fluid center-block w-100">
                <div class="caption text-center">
                <h6>Multivariable Moran Statistics
                </h6>
                </div>
            </a>
        </div>
        <div class="col-6">
        <a href="https://nbviewer.org/github/pysal/esda/blob/main/notebooks/localjoincounts.ipynb" class="thumbnail">
                <img src="_static/images/joincount.png" class="img-fluid center-block w-100">
                <div class="caption text-center">
                <h6>Discrete Spatial Autocorrelation with Join Counts
                </h6>
                </div>
            </a>
        </div>
      </div>
    </div>


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   Installation <installation>
   API <api>
   Tutorial <tutorial>
   References <references>

************
Introduction
************

**esda** implements measures for the exploratory analysis spatial data and is part of the  `PySAL family <https://pysal.org>`_

Details are available in the `esda api <api.html>`_.


***********
Development
***********

esda development is hosted on github_.

.. _github : https://github.com/pysal/esda

Discussions of development occurs on the
`developer list <http://groups.google.com/group/pysal-dev>`_
as well as the `esda Discord channel <https://discord.gg/Re46DjyB9U>`_.

****************
Getting Involved
****************

If you are interested in contributing to PySAL please see our
`development guidelines <https://github.com/pysal/pysal/wiki>`_.


***********
Bug reports
***********

To search for or report bugs, please see esda's issues_.

.. _issues :  http://github.com/pysal/esda/issues


***********
Citing esda
***********

If you use PySAL-esda in a scientific publication, we would appreciate citations to the following paper:

  `PySAL: A Python Library of Spatial Analytical Methods <http://journal.srsa.org/ojs/index.php/RRS/article/view/134/85>`_, *Rey, S.J. and L. Anselin*, Review of Regional Studies 37, 5-27 2007.

  Bibtex entry::

      @Article{pysal2007,
        author={Rey, Sergio J. and Anselin, Luc},
        title={{PySAL: A Python Library of Spatial Analytical Methods}},
        journal={The Review of Regional Studies},
        year=2007,
        volume={37},
        number={1},
        pages={5-27},
        keywords={Open Source; Software; Spatial}
      }


*******************
License information
*******************

See the file "LICENSE.txt" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   Installation <installation>
   Tutorial <tutorial>
   API <api>
   References <references>


.. _PySAL: https://github.com/pysal/pysal
