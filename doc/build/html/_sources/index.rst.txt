..  _main:

**pypce**
=========

**pypce** is a pure Python Uncertainty Quantification (UQ) software library.
[#]_. It is built with object oriented design principals in mind and is thus
modular. 

The pypce library contains modules for building multivariate polynomial
regression models (built with the scikit-learn API in mind), constructing
quadrature rules for high-dimensional numerical integration, performing
sensitivity analysis or feature importance, preprocessing utilities for
transforming feature and target spaces, e.g. PCA, and custom pipelines for
fitting multi-target regression models and comparing different machine learning
(ML) methods.

The code is easy to install. After downloading, make sure you also have
``numpy`` and ``scikit-learn``. Then, cd into the ``pypce`` directory, i.e. the
folder with the ``setup.py`` file, and run

::

	pip install .

You can also run a suite of unittests with 

::

   python -m pytest -v -s pypce/tests

That's it. Now you are ready to use **pypce**. 

.. toctree::
   :maxdepth: 1
   
   getting_started
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [#] Python 3+ is required and pypce has only been tested for 3.7.6 so far. 