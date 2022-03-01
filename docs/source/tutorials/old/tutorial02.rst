Some other usage cases
======================

In this tutorial, we will look at some uncommon usage cases for the pce
regression classes. For example, we will look at how to construct a pce
with a given coefficient array, using normalized Legendre bases, using
different fit algorithms with custom fit parameters, etc. Again, let us
import the relavant libraries.

.. code-block:: ipython3
    :linenos:

    import pypce
    import numpy as np
    import matplotlib.pyplot as mpl

And let us create some sample data with and without noise this time,
which will necessitate the use of a regularization method.

.. code-block:: ipython3
    :linenos:

    rn = np.random.RandomState(113)
    X = 2*rn.rand(100,2)-1
    y = X[:,0] + .5*(3*X[:,1]**2-1) 
    y_w_noise = y.copy() + .1*rn.randn(len(y))
    print(X.shape,y.shape,y_w_noise.shape)


.. parsed-literal::

    (100, 2) (100,) (100,)


And let’s plot the input output univariate pairs with noise.

.. code-block:: ipython3
    :linenos:

    [mpl.plot(X[:,i],y_w_noise,'.') for i in range(X.shape[1])]

Defining custom coefficient array
---------------------------------

First, suppose we already have a coefficient array we want to predict
with. This is not common, but the pce regression class has that as an
option. In this case you initial the pce object but have to make sure
the coefficient array is the same size as the number of basis functions,
which is equivalent to the number of rows of the multiindex array.

.. code-block:: ipython3
    :linenos:

    pce_ref = pypce.PCEReg(order=2)
    pce_ref.compile(dim=2) # must run this to generate multiindex
    print("multiindex:\n",pce_ref.mindex)
    print("number of basis elements: ", pce_ref.mindex.shape[0])


.. parsed-literal::

    multiindex:
     [[0 0]
     [1 0]
     [0 1]
     [2 0]
     [1 1]
     [0 2]]
    number of basis elements:  6


Here, we said we wanted a second order polynomial and used the compile
method from the PCEBuilder class to construct the multiindex. We were
then able to print out the multiindex shape which, in this case, has 6
basis elements. So we need a coefficient array of size 6.

.. code-block:: ipython3
    :linenos:

    coef = np.zeros(6)
    coef[1] = 1.0; coef[-1] = 1.0

Let us construct a new pce regression object with this coefficient
array.

.. code-block:: ipython3
    :linenos:

    pce = pypce.PCEReg(order=2,coef=coef)
    pce.compile(dim=2)
    pce.feature_importances()




.. parsed-literal::

    array([0.625, 0.375])



We can now make predictions with this polynomial object. We can bypass
the fitting because we already know what the coefficient array is.

.. code-block:: ipython3
    :linenos:

    ypred = pce.predict(X)
    print("prediction error = ", np.sum(ypred-y))

Note what happens if the coefficient array is the wrong size. The
initialization is fine, but the predict method fails.

.. code-block:: ipython3
    :linenos:

    pce = pypce.PCEReg(order=2,coef=coef[:-1])
    pce.predict(X)

Custom multiindex array
-----------------------

We can also build a pce object with a custom multiindex array as well.
Let’s use the multiindex of the reference pce above and modify it.

.. code-block:: ipython3
    :linenos:

    custom_mindex = pce_ref.mindex[[1,-1]]
    print(custom_mindex)


.. parsed-literal::

    [[1 0]
     [0 2]]


Here we just took the second and last row of the multiindex. Now let’s
construct our pce with this.

.. code-block:: ipython3
    :linenos:

    pce = pypce.PCEReg(customM=custom_mindex)
    pce.fit(X,y)




.. parsed-literal::

    PCEReg(coef=array([1., 1.]), customM=array([[1, 0],
           [0, 2]]), order=None)



Note that the custom multiindex has to have the right dimensions. That
is, the custom_mindex.shape[1] has to match the dimension of X,
i.e. X.shape[1].

.. code-block:: ipython3
    :linenos:

    pce.feature_importances_




.. parsed-literal::

    array([0.625, 0.375])



Custom multiindex and coefficient array
---------------------------------------

We can also feed it both a custom multiindex AND coefficient array.

.. code-block:: ipython3
    :linenos:

    coef = np.array([1.0,1.0])
    pce = pypce.PCEReg(customM=custom_mindex,coef=coef)
    ypred = pce.predict(X)
    print("error is ", np.sum(ypred-y))

Again, we can bypass the predict function since we already know the
coefficient array.

Using normalized Legendre basis
-------------------------------

Another option is to use a normalized Legendre basis. The current method
actually computes the normalization scalar, but in future versions, we
will just use a normalized legendre function as input. Let’s see how it
works anyway.

.. code-block:: ipython3
    :linenos:

    pce = pypce.PCEReg(order=2,fit_type='linear',normalized=True)
    pce.fit(X,y)

Notice how the coefficient values have changed. This is due to the
normalization constant. While the coefficient array changed, the actual
polynomials are just scaled versions of themselves, so the prediction
should be exactly the same, as well as the feature importances. Let’s
see.

.. code-block:: ipython3
    :linenos:

    ypred = pce.predict(X)
    print("mse = {0:E}".format(np.sum((ypred-y)**2)))

.. code-block:: ipython3
    :linenos:

    pce.feature_importances_
