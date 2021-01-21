..  _api:

API
===

.. MultiIndex Classes
.. ------------------

.. .. automodule:: pypce.multiindex
.. 	:members:

PCE Base Class
--------------

.. autoclass:: pypce.PCEBuilder
	:members:
	:exclude-members: get_params, set_params, computeNormSq

PCE Regression Class
--------------------

.. autoclass:: pypce.PCEReg
	:members: 
	:exclude-members: get_params, set_params, compile, eval, polyeval, score, fit_transform, computeMoments, computeSobol, computeNormSq, sensitivity_indices, multiindex
	:private-members: _compile, _quad_fit

.. Preprocessing Utilities
.. -----------------------

.. .. automodule:: pypce.preprocessing
.. 	:members:

.. Quadrature
.. -----------------------

.. .. autoclass:: pypce.QuadGen
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

