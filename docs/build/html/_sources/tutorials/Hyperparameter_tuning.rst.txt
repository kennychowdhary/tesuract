Let’s try some hyper-parameter tuning for our polynomial fitting before
we give up!

Hyper-parameter tuning
======================

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

Grid search tuning
------------------

There are essentially three parameters to tune in the polynomial
regression class. The first, and the most obvious, is the polynomial
order, which has the keyword ``order`` in the constructor. The next is
the type of polynomial interaction terms called ``mindex_type``.
“total_orer” os the default, but an alternative is “hyperbolic” which
has even fewer interaction terms which emphasizes more higher-order
terms. In practice, this rarely leads to a better polynomial, but we can
try it anyway. Last, but not least, there is the polynomial
``fit_type``, which determines the solver used to solve the least
squares problem (Note even though polynomials are non-linear, the
fitting boils down to a linear problem). This can be a bunch of
different algorithms, but the three most widely used are ‘linear’,
‘LassoCV’, and ‘ElasticNetCV’.

With these parameters in mind, we create a parameter grid just like one
would when using the GridSearchCV method in sklearn.

.. code:: ipython3

    pce_grid = {
        'order': list(range(1,12)),
        'mindex_type': ['total_order','hyperbolic'],
        'fit_type': ['linear','ElasticNetCV','LassoCV'],
        }

Now we use the regression wrapper CV class which wraps the PCEReg class
in sklearn’s grid search CV functionality.

.. code:: ipython3

    # hyper-parameter tune the PCE regression class using all available cores
    pce = tesuract.RegressionWrapperCV(
        regressor='pce',
        reg_params=pce_grid,
        n_jobs=-1,
        scorer='r2')
    pce.fit(X,y)
    print("Hyper-parameter CV PCE score is {0:.3f}".format(pce.best_score_))


.. parsed-literal::

    Fitting 5 folds for each of 66 candidates, totalling 330 fits
    Fitting 5 folds for each of 66 candidates, totalling 330 fits
    Hyper-parameter CV PCE score is 0.999


Why so many fits? For each k-fold (5 total) we have to compute 66 fits
corresponding to 66 different parameter combinations. This repeats five
times to get an average cross validation score.

Look at that! We got all the way to an R2 score of basically 1! How did
we do that? One of our parameter combinations must have been really
good. Which one was it? We can easily find out by called the
best_params\_ attribute.

.. code:: ipython3

    pce.best_params_




.. parsed-literal::

    {'fit_type': 'ElasticNetCV', 'mindex_type': 'total_order', 'order': 7}



So it seems like 8th order way too high and probably overfit, so a
fourth order was much better. Elastic net regularization also seemed to
work the best, which uses a mix of l1 and l2 regularization.

We can also extract the best scores, and the best estimator, i.e a
PCEReg object with the fitted coefficients.

.. code:: ipython3

    pce_best = pce.best_estimator_

Now, to be fair, we probably should hyper-parameter tune the MLP
regressor to perform a completely fair comparison, and it may probably
give us ultimately a better model. In general however, neural networks
are much hard to hyper-parameter tune and take longer to train, so the
polynomial model can be preferred when accuracy and simplicity is
required.
