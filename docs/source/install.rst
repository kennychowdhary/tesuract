..  _install:


Installation
============

The code is easy to install with ``pip``. Simply run 

.. code-block:: bash
   
   pip install tesuract

and this will install the pypi version of tesuract. For the latest development build, just clone and install the repo. Make sure you have ``numpy`` and
``scikit-learn``. Then

.. code-block:: bash

   git clone git@github.com:kennychowdhary/tesuract.git
   cd tesuract

and simple run

.. code-block:: bash

   pip install .


You can also run a suite of unit tests and regression tests before installation by typing 

::

   python -m pytest -v -s tesuract/tests

to check that the library works. Note that the `python -m` allows you to automatically add the current path to the Python path, without having to change any environmental variables. That's it! Now you are ready to use **tesuract**. 

