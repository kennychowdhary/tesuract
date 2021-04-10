import numpy as np

# cross validation test
def multi_target_rel_l2_error(y_true,y_pred):
	if y_true.ndim == 2 and y_true.ndim == 2: 
	    error_array =  np.linalg.norm(y_true - y_pred,axis=1)/np.linalg.norm(y_true,axis=1)
	    return error_array.mean()
	else:
		return np.linalg.norm(y_true - y_pred)/np.linalg.norm(y_true)