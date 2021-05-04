..  _api:

API
===

.. MultiIndex Classes
.. ------------------

.. .. automodule:: tesuract.multiindex
.. 	:members:

PCE Base Class
--------------

.. autoclass:: tesuract.PCEBuilder
	:members:
	:exclude-members: get_params, set_params, computeNormSq

PCE Regression Class
--------------------

.. autoclass:: tesuract.PCEReg
	:members: 
	:exclude-members: get_params, set_params, compile, eval, polyeval, score, fit_transform, computeMoments, computeSobol, computeNormSq, sensitivity_indices, multiindex
	:private-members: _compile, _quad_fit

.. Preprocessing Utilities
.. -----------------------

.. .. automodule:: tesuract.preprocessing
.. 	:members:

.. Quadrature
.. -----------------------

.. .. autoclass:: tesuract.QuadGen
.. 	:members:

.. The ``shape`` module
.. --------------------

.. .. automodule:: trianglelib.shape
.. 	:members:
.. 	:member-order: bysource

.. The ``utils`` module
.. --------------------

.. .. automodule:: trianglelib.utils
.. 	:members:

