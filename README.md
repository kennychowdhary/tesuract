![License](https://img.shields.io/github/license/kennychowdhary/tesuract)
![PyPI](https://img.shields.io/pypi/v/tesuract)


<!-- ![image](docs/source/_static/tesuract_logo.png) -->

# <img src="docs/source/_static/tesuract_logo.png" alt="drawing" width="250"/> 

<!-- **tesuract** -->

**tesuract**, which stands for **te**nsor **sur**rogate **a**utomation
and **c**ompu**t**ation, is a software library to perform automated
supervised (and semi-unsupervised via data-driven reduced order
modeling) machine learning tasks with single and multi-target output
data.[^1] One of the key features is that it is fully compatible with
scikit-learn\'s API, e.g., their *set, fit, predict* functionality,
allowing flexibility and modularity. It also contains tools to quickly
and easily implement multi-variate Legendre polynomial (the original
universal approximator!) regression models.

## Documentation

Please see the full documentation [here](https://kennychowdhary.github.io/tesuract/build/html/index.html).

## Installation

The code is easy to install with `pip`. Simply run

``` bash
pip install tesuract
```

and this will install the pypi version of tesuract. For the latest
development build, just clone and install the repo. Make sure you have
`numpy` and `scikit-learn`. You might also need the `alive-progress` bar
library. Then

``` bash
git clone git@github.com:kennychowdhary/tesuract.git
cd tesuract
```

and simple run

``` bash
pip install .
```

You can also run a suite of unit tests and regression tests before
installation by typing

    python -m pytest -v -s tesuract/tests

to check that the library works. Note that the [python -m]{.title-ref}
allows you to automatically add the current path to the Python path,
without having to change any environmental variables. That\'s it! Now
you are ready to use **tesuract**.

## Usage/ Quickstart

Let\'s see how easy it is to create a multivariate polynomial regression
model. Let\'s create a $4^{th}$ order polynomial regression model[^2] on
the $[-1,1]^5$ hypercube, using sklearn\'s LassoCV ($\ell_1$ sparsity
constraint) fitting.[^3]

``` python
import tesuract
from sklearn.datasets import make_friedman1

X,y = make_friedman1(n_samples=100,n_features=5)
pce = tesuract.PCEReg(order=8, fit_type='LassoCV') # create an 8th order polynomial
pce.fit(X,y)
```

That\'s it![^4] You\'ve fit your first polynomial chaos expansion (PCE)
using tesuract (with a linear least squares solver). You can try
changing the type of solver, e.g., LassoCV or ElasticNetCV, getting
feature importances, etc.

[^1]: Python 3+ is required and tesuract has been tested for 3.7.6 and
    3.8.10 so far.

[^2]: In short, a $4^{th}$ order polynomial means that the terms are no
    higher than an $x_i^4$ for each dimension.

[^3]: Don\'t worry! There are a lot more customization options that we
    will get into later

[^4]: The dimensionality is automatically determined by looking at the
    size of the data matrix columns.

