Here are some of the most common uses for tesuract. This is not
comprehensive, but merely highlights the strengths and flexibility of
tesuract.

Feature Importances
===================

Setup
-----

.. code:: ipython3

    # First, import the libraries we will use
    import numpy as np
    import tesuract
    import matplotlib.pyplot as plt

.. code:: ipython3

    # import a data set for our regression problem
    from sklearn.datasets import make_friedman1
    X,y = make_friedman1(n_samples=100,n_features=5)

.. code:: ipython3

    # rescale the input
    X = 2*X - 1
    # center and scale the output as well for good measure (not required)
    y = (y - y.mean())/np.sqrt(np.var(y))

Total order Sobol sensitivities
-------------------------------

One last thing, before we move onto more advanced usage cases, in
particular, tensor surrogates. With orthogonal polynomials like in PCEs,
we can (almost) automatically obtain feature importances in the form of
Sobol sensitivity indices. In particular, we can call feature
importances to obtain normalized (summed to one) Sobol total order
indices. Let’s run the model at the best parameters and extract the
Sobol indices/ feature importances.

Now we use the regression wrapper CV class which wraps the PCEReg class
in sklearn’s grid search CV functionality.

.. code:: ipython3

    pce_best = tesuract.PCEReg(order=4,fit_type='ElasticNetCV')
    pce_best.fit(X,y);

.. code:: ipython3

    # compute the normalized (sum to 1) Sobol total order indices
    pce_best.feature_importances_




.. parsed-literal::

    array([0.25483936, 0.25010487, 0.09265639, 0.32062786, 0.08177152])



Now technically, the Sobol total order indices shouldn’t be normalized,
but we do it for consistency and with only some loss of generality. For
the original total order indices use ``computeSobol()`` method.

.. code:: ipython3

    pce_best.computeSobol()




.. parsed-literal::

    [0.275399294692865,
     0.27028283968131295,
     0.10013172012221548,
     0.3464954852447724,
     0.08836868613421243]



The larger the value, the more “important” the parameter is. So, the
first, second and fourth parameters are more importance features in the
model than the remaining two.
