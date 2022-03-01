Here are some of the most common uses for tesuract. This is not
comprehensive, but merely highlights the strengths and flexibility of
tesuract.

Fitting with sklearn’s API
==========================

Here we show how tesuract can be used to construct (and easily fit) a
Legendre polynomial) to data for more than just a single-dimension. This
highlights the ease to which we can create complex interacting
polynomial functions that scale well for higher dimensions, which cannot
be done without some care in say scikit-learn using the polynomial
feature library.

.. code:: ipython3

    # First, import the libraries we will use
    import numpy as np
    import tesuract
    import matplotlib.pyplot as plt

We will create a five-dimensional regression problem with 100 samples.
Thus, this will not be as trivial as a one-dimensional polynomial
fitting. In this case, no finite polynomial will return an exact
solution so this will be a little bit of a challenge.

.. code:: ipython3

    # import a data set for our regression problem
    from sklearn.datasets import make_friedman1
    X,y = make_friedman1(n_samples=100,n_features=5)

For the polynomial regression, it is not required, but recommended, to
normalize the input X data to :math:`[-1,1]`. Since the
``make_friedman1`` function is on the unit interval, this is very easy.
Once we do this, we can now fit using a polynomial. See the later
tutorials on how to do this without having to rescale the input.

.. code:: ipython3

    # rescale the input
    X = 2*X - 1
    # center and scale the output as well for good measure (not required)
    y = (y - y.mean())/np.sqrt(np.var(y))

Since the regression test problem is in fact a sinusoidal function, let
us try a higher order polynomial. Now, in order to create a
five-dimensional polynomial using the tools already available in
sklearn, we would have to create a one-dimensional monomial feature and
then, in the simplest way, we can take a full tensor product of all
interaction for each dimension. This means that if we have an 8th order
polynomial in each of the five dimensions (each with 9 coefficient
including the bias term), we would have :math:`9^5 = 59,049`
coefficients to learn! This would lead to terms that have a total order
(sum of exponents) of :math:`8\times5 = 40`, which is not ideal. We can
limit the number of interaction terms by using Smolyak type expansions
from uncertainty quantification, a.k.a., polynomial chaos expansions. In
this case, if we want a total order of 8, then the total number of terms
is only :math:`1,287`. This is all done under the hood, and in a later
tutorial we will explain what’s going on. But for now, just trust me!

.. code:: ipython3

    # create an 8th order polynomial (total order amongst all dimensions)
    pce = tesuract.PCEReg(order=8)
    pce.fit(X,y)




.. parsed-literal::

    PCEReg(coef=array([-0.00903059,  0.44826357,  0.64919499, ...,  0.06610481,
            0.04319052,  0.06408081]),
           order=8)



And that’s it. We just used the default linear least squares estimator
to fit the data with an 8th order five-dimensional polynomial. Just to
show you this is actually quite complex, we can show what’s called the
multi-index set, which represents all the interaction terms between the
orthogonal one-dimensional Legendre polynomial.

.. code:: ipython3

    pce.mindex




.. parsed-literal::

    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           ...,
           [0, 0, 1, 0, 7],
           [0, 0, 0, 1, 7],
           [0, 0, 0, 0, 8]])



So for example, the second-to-last multi-index corresponds to a
five-dimensional polynomial basis function (like a Fourier basis but
with polynomials) as the product of the first order 1-D Legendre
polynomial of the fourth dimension and the 8th order 1-D Legendre
polynomial of the fifth dimenion.

Now that we know how to create a multi-variate polynomial expansion or
Polynomial Chaos Expansion (PCE), how to do we evaluate it’s accuracy.
Here is where tesuract really shine!

The PCEReg class is actually an sklearn estimator class. This means it
can be fully integrated in the sklearn universe. In particular, we can
wrap the PCEReg class into say the cross_val_score function in sklearn.
This computes the k-fold cross validation score of any sklearn
estimator. Let’s try this to get a score.
