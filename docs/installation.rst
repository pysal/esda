.. Installation

Installation
============

esda supports Python `>=3.6`. Please make sure that you are
operating in a Python 3 environment.

Installing released version
---------------------------

esda is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U esda


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .


You may also install the latest stable esda via conda-forge channel by running::

  conda install --channel conda-forge esda



Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of esda on github - `pysal/esda`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/esda`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/esda.git

You can  also `fork`_ the `pysal/esda`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/esda`_, you can
contribute to esda development.

.. _Python Package Index: https://pypi.org/project/esda/
.. _pysal/esda: https://github.com/pysal/esda
.. _fork: https://help.github.com/articles/fork-a-repo/


