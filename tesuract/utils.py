import numpy as np

# cross validation test
def multi_target_rel_l2_error(y_true,y_pred):
	if y_true.ndim == 2 and y_true.ndim == 2: 
	    error_array =  np.linalg.norm(y_true - y_pred,axis=1)/np.linalg.norm(y_true,axis=1)
	    return error_array.mean()
	else:
		return np.linalg.norm(y_true - y_pred)/np.linalg.norm(y_true)


def plot_samples(Y,x=None,ax=None,q=[.005,.5,.995],show_mean=True,label=None,color=None,xlabel='',ylabel='',fs=10,alpha=.2,rotation=None,labelpad=10):
	assert Y.ndim == 2, "Y must contain samples and have 2 dimensions."
	nsamples, dim = Y.shape
	mu = np.mean(Y,axis=0)
	var = np.var(Y,axis=0)
	Q = np.quantile(Y,q=q,axis=0)
	if ax is None:
		fig,ax = mpl.subplots()
	if x is None:
		x = list(np.arange(dim))
	q_label = r"{}".format(label + ' 99% Q')
	# mu_label = r"{}".format(label + ' median')
	mu_label = None
	if q is None:
		ax.fill_between(x, mu - np.sqrt(var), mu + np.sqrt(var),color=color, alpha=alpha,label=label)
		if show_mean:
			ax.plot(x, mu, '--',color=color, lw=1.5, alpha=0.5,label=mu_label)
	else:
		ax.fill_between(x, Q[0], Q[2],color=color, alpha=0.2,label=q_label)
		if show_mean:
			ax.plot(x, Q[1], '--',color=color, lw=1.5, alpha=0.5,label=mu_label)

	ax.set_xlabel(r"{}".format(xlabel),fontsize=fs)
	ax.set_ylabel(r"{}".format(ylabel),fontsize=fs,rotation=rotation,labelpad=labelpad)
	return ax