..  _main:

**tesuract**
============

**tesuract**, which stands for **te**\ nsor **sur**\ rogate **a**\ utomation and **c**\ ompu\ **t**\ ation, is a software library to perform automated supervised (and semi-unsupervised via data-driven reduced order modeling) machine learning tasks with single and multi-target output data.\ [#]_ One of the key features is that it is fully compatible with scikit-learn's API, e.g., their *set, fit, predict* functionality, allowing flexibility and modularity. It also contains tools to quickly and easily implement multi-variate Legendre polynomial (the original universal approximator!) regression models. 

Installation
------------

The code is super easy to install with ``pip``. Make sure you have ``numpy`` and
``scikit-learn``. Then, after downloading, cd into the ``tesuract`` directory, i.e.
the folder with the ``setup.py``, and run

.. code-block:: python

	pip install .


You can also run a suite of unit tests and regression tests before installation with 

::

   python -m pytest -v -s tesuract/tests

to check that the library works. That's it! Now you are ready to use **tesuract**. 

Quickstart
----------------

Let's see how easy it is to create a multivariate polynomial regression model. 
Let's create a :math:`4^{th}` order polynomial regression model\ [#]_ on the 
:math:`[-1,1]^5` hypercube, using sklearn's LassoCV (:math:`\ell_1` sparsity constraint) fitting.\ [#]_

.. code-block:: python

   import tesuract
   from sklearn.datasets import make_friedman1

   X,y = make_friedman1(n_samples=100,n_features=5)
   pce = tesuract.PCEReg(order=8, fit_type='LassoCV') # create an 8th order polynomial
   pce.fit(X,y)

That's it! [#]_ You've fit your first polynomial chaos expansion (PCE) using tesuract (with a linear least squares solver). You can try changing the type of solver, e.g., LassoCV or ElasticNetCV, getting feature importances, etc. 

.. toctree::
   :maxdepth: 4
   :caption: User Guide
   
   introduction
   guide
   api

.. .. toctree::
..    :maxdepth: 2
..    :caption: Tutorials
   
..    guide



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [#] Python 3+ is required and tesuract has only been tested for 
         3.7.6 so far. 

.. [#] In short, a :math:`4^{th}` order polynomial means that the
       terms are no higher than an :math:`x_i^4` for each dimension. 

.. [#] Don't worry! There are a lot more customization options that we will get into later

.. [#] The dimensionality is automatically determined by looking at the size 
      of the data matrix columns. 