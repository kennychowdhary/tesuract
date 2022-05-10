05-05-2022 v 0.2.0 changes
==========================

Fix RegressionWrapperCV where it was using grid search twice
 - simplified fit function so that fit_multiple_reg is only being used (easier to debug)

added new multi regression wrapper tests to make sure different ways of calling tesuract works as expected
 - tests for MRegressionWrapperCV and RegressionWrapperCV and for the compute_cv_score function

Added a pytest.ini file with new markers for "unit" testing and "regression" testing
 - pytest -m "unit" tesuract/tests/ for unit testing only 
 - pytest -m "regression" tesuract/tests/ for regression testing only

changed np.int to int in multi-index module

removed check that regression parameters for grid search has to be the same length as the regression list

regression list is automatically interpreted as a list if it is a string

compute_cv_score function for MRegressionWrapperCV which computes the full cv score

