..  _intro:

Main features
=============

tesuract's main feature is constructing multivariate polynomial regression models
using classic orthogonal polynomial constructions, e.g. Legendre polynomials. We
use the scikit-learn API to build these estimators, which allows easy and seamless integration with the scikit-learn. Here is a list of the main features of this library. 

* **Integration with scikit-learn**
    Both :mod:`~tesuract.PCEBuilder` and :mod:`~tesuract.PCEReg` classes for
    constructing multivariate polynomial interpolation inherit the BaseEstimator class from scikit-learn. This means both these object integrate seamlessly with the scikit-learn environment, which, for example, allows one to use the grid search cross-validation wrapper or even third party libraries like skopt on top of these polynomial estimator classes. 

* **Multivariate polynomial regression**
    We provide an object oriented polynomial constructor class that creates a
    polynomial estimator (based on the scikit-learn estimator API) and a
    corresponding multivariate polynomial regression class that utilizes
    existing linear regression algorithms in scikit-learn to train the model. 

* **Hyper-parameter search for best model fit**
    
    tesuract provides methods to perform hyper-parameter search for finding the
    best fit polynomial model. Moreover, we provide methods to compare these
    methods to other popular machine learning regression models like random
    forest regression and multi-layer perceptron models. [#]_ 

* **Variance based sensitivity analysis**

    The polynomial regression class allows for easy feature importance or
    sensitivity analysis. We use the Sobol sensitivity index to compute which
    features are more important.

* **Sparse quadrature methods for high-dimensional integration**

    An alternate way to estimate the training weights of the multivariate
    polynomial model (i.e., the coefficients of the polynomial expansion), is
    direct numerical integration. Due to the orthogonality of the polynomial
    basis terms, the training weights can be written in analytic form as a
    integration rule. We provide high-dimensional, sparse, integration rules 
    to estimate these coefficients. This is useful for *smooth* functions, rather
    than functions corrupted by noise. 

* **Utilities for preprocessing**
    While scikit-learn contains many preprocessing utilities, we add a few more
    that may be more tailored for scientific computing applications. Utilities
    include min-max transforms for multi-target outputs, transforms for scaling
    domains with known physical bounds, and more customizable dimension reduction
    transforms, e.g., PCA. 

* **Pipelines for multi-target fitting** [#]_

  

.. [#] In the future, we will be able to compare methods in tensorflow. The machinery is there for connecting tensorflow to sklearn, but it just needs to be done!

.. [#] This is an advanced usage for tesuract.

.. Special triangles
.. -----------------

.. There are two special kinds of triangle
.. for which trianglelib offers special support.

.. *Equilateral triangle*
.. 	All three sides are of equal length.

.. *Isosceles triangle*
.. 	Has at least two sides that are of equal length.

.. These are supported both by simple methods
.. that are available in the :mod:`trianglelib.utils`,
.. and also by a pair of methods of the main
.. Triangle class itself.

.. Triangle dimensions
.. -------------------

.. The library can compute triangle perimeter, area,
.. and can also compare two triangles for equality.
.. Note that it does not matter which side you start with,
.. so long as two triangles have the same three sides in the same order!

.. >>> from trianglelib.shape import Triangle
.. >>> t1 = Triangle(3, 4, 5)
.. >>> t2 = Triangle(4, 5, 3)
.. >>> t3 = Triangle(3, 4, 6)
.. >>> print t1 == t2
.. True
.. >>> print t1 == t3
.. False
.. >>> print t1.area()
.. 6.0
.. >>> print t1.scale(2.0).area()
.. 24.0

.. Valid triangles
.. ---------------

.. Many combinations of three numbers cannot be the sides of a triangle.
.. Even if all three numbers are positive instead of negative or zero,
.. one of the numbers can still be so large
.. that the shorter two sides
.. could not actually meet to make a closed figure.
.. If c is the longest side, then a triangle is only possible if:

.. ::

.. 	a + b > c

.. While the documentation
.. for each function in the utils module
.. simply specifies a return value for cases that are not real triangles,
.. the Triangle class is more strict
.. and raises an exception if your sides lengths are not appropriate:

.. ::

.. 	>>> from trianglelib.shape import Triangle
.. 	>>> Triangle(1, 1, 3)
.. 	Traceback (most recent call last):
.. 	  ...
.. 	ValueError: one side is too long to make a triangle

.. If you are not sanitizing your user input
.. to verify that the three side lengths they are giving you are safe,
.. then be prepared to trap this exception
.. and report the error to your user.
