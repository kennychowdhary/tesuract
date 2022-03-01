Let’s perform a model comparison with some of the other sklearn
estimators. Since the PCE regression class is of the same type, we can
feed it directly

Model comparison
================

Setup
~~~~~

.. code:: ipython3

    # First, import the libraries we will use
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

Cross-validation score
----------------------

.. code:: ipython3

    # compute the cross validation score (5-fold by default)
    # of the pce we constructed earlier, i.e. 8th order linear fit
    from sklearn.model_selection import cross_val_score
    pce = tesuract.PCEReg(order=8)
    pce_score = cross_val_score(pce,X,y,scoring='r2').mean()
    print("PCE score is {0:.3f}".format(pce_score))


.. parsed-literal::

    PCE score is 0.836


Not bad for a first pass. How does it compare to something like random
forests regression or MLPs? Now, we can compare apples to applies within
the same environment since these models are all part of the sklearn
base-estimator class.

.. code:: ipython3

    # Let's try a simple random forest estimator
    from sklearn.ensemble import RandomForestRegressor
    rfregr = RandomForestRegressor(max_depth=5,n_estimators=100)
    rf_score = cross_val_score(rfregr,X,y,scoring='r2').mean()
    print("RF score is {0:.3f}".format(rf_score))


.. parsed-literal::

    RF score is 0.685


.. code:: ipython3

    # Let's try a simple 4-layer neural network (fully connected)
    from sklearn.neural_network import MLPRegressor
    mlpregr = MLPRegressor(hidden_layer_sizes=(100,100,100,100))
    mlp_score = cross_val_score(mlpregr,X,y,scoring='r2').mean()
    print("MLP score is {0:.3f}".format(mlp_score))


.. parsed-literal::

    MLP score is 0.939


Wow! So the MLP way out-performed the 8th order polynomial with a linear
fit! But wait. What if we tried a different polynomial order? Or a
different fitting procedure like a sparse l-1 solver? Can we leverage
the hyper-parameter tuning that sklearn has? Yes! Moreso, we created an
easy wrapper for the grid search cv functionality and a new pce
regression wrapper that has cross-validation and hyper-parameter tuning
built in!
