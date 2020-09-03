import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import pypce
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import KFold, cross_val_score, cross_validate

from sklearn import linear_model
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

import pandas as pd
import ipdb

n, d = 30, 2
rn = np.random.RandomState(45)
X = 2 * rn.rand(n, d) - 1
y = 3 + 2 * .5 * (3 * X[:, 0] ** 2 - 1) + X[:, 1] * .5 * (3 * X[:, 0] ** 2 - 1)
y2 = 1 + 2 * .5 * (3 * X[:, 1] ** 2 - 1) + 3 * X[:, 0] * .5 * (3 * X[:, 1] ** 2 - 1)
Y = np.vstack([y, y2]).T  # test for multioutput regressor
c_true = np.array([3., 0., 0., 2., 0., 0., 0., 1., 0., 0., ])


class MultiOutputCV(BaseEstimator, RegressorMixin):
    def __init__(self, regressor_model, regressor_param_grid,
                 score_metric='neg_root_mean_squared_error',
                 n_jobs=4,cv=None):
        self.regressor_model = regressor_model
        self.regressor_param_grid = regressor_param_grid
        self.score_metric = score_metric
        self.n_jobs_ = n_jobs
        self.cv = cv
    def _setupCV(self, shuffle=False, randstate=13):
        if self.cv == None:
            self.cv = KFold(n_splits=5)  # ,shuffle=True,random_state=13)
    def fit(self, X, Y):
        self.n, self.dim = X.shape
        if Y.ndim == 1: Y = np.atleast_2d(Y).T
        assert len(Y) == self.n, "mistmatch in no of samples btwn X and Y."
        self.ntargets = Y.shape[1]
        self._setupCV()
        pceCV = GridSearchCV(self.regressor_model, self.regressor_param_grid, scoring=self.score_metric,
            n_jobs=self.n_jobs_, cv=self.cv, 
            verbose=1, return_train_score=True)
        self.best_estimators_ = []
        self.best_params_ = []
        # t = tqdm(range(self.ntargets), desc='target #', leave=True)
        # for i in t:
        for i in range(self.ntargets):
            # t.set_description("target #%i" % (i + 1))
            pceCV.fit(X, Y[:, i])
            self.best_estimators_.append(pceCV.best_estimator_)
            self.best_params_.append(pceCV.best_params_)
        self.pceCV = pceCV
        return self

    def predict(self, X):
        y = []
        for estimator_ in self.best_estimators_:
            y.append(estimator_.predict(X))
        if self.ntargets == 1:
            y = y[0]
        else:
            y = np.vstack(y).T
        return y

    def feature_importances(self):
        S = []
        for estimator_ in self.best_estimators_:
            assert hasattr(estimator_,'feature_importances_')
            S.append(estimator_.sensitivity_indices())
        if self.ntargets == 1:
            S = np.array(S[0])
        else:
            S = np.vstack(S)
        return S

class PCATargetTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, cutoff=1e-2, svd_solver='arpack'):
        self.n_components = n_components
        self.cutoff = cutoff
        self.svd_solver = svd_solver
        self.K = None

    def fit(self, Y):
        self.n, self.d = Y.shape
        if self.n_components == 'auto':
            self._compute_K(Y)
        if isinstance(self.n_components, int):
            self.K = self.n_components
            if self.n_components > self.d:
                warnings.warn("No of components greater than dimension. Setting to maximum allowable.")
                self.n_components = self.d
        assert self.K is not None, "K components is not defined or being set properly."
        self.pca = PCA(n_components=self.K, svd_solver=self.svd_solver)
        self.pca.fit(Y)
        return self

    def fit_transform(self, Y):
        self.fit(Y)
        return self.pca.transform(Y)

    def transform(self, Y):
        assert hasattr(self, 'pca'), "Perform fit first."
        return self.pca.transform(Y)

    def inverse_transform(self, Yhat):
        return self.pca.inverse_transform(Yhat)

    def _compute_K(self, Y, max_n_components=50):
        ''' automatically compute number of PCA terms '''
        self.pca = PCA(n_components=min(max_n_components, self.d), svd_solver=self.svd_solver)
        self.pca.fit(Y)
        self.cumulative_error = np.cumsum(self.pca.explained_variance_ratio_)
        self.K = np.where(1 - self.cumulative_error <= self.cutoff)[0][0]

def plot_feature_importance(S,feature_labels,extra_line_plot=None):
    assert isinstance(S,np.ndarray), "S must be a numpy array"
    if S.ndim == 1:
        ntargets = 1
        ndim = len(S)
        S = np.atleast_2d(S)

    ntargets, ndim = S.shape
    # normalize across columns (if not already)
    S = S / np.atleast_2d(S.sum(axis=1)).T

    # plot sobol indices as stacked bar charts

    import matplotlib.pyplot as plt
    import matplotlib._color_data as mcd
    import seaborn as sns

    xkcd_colors = []
    xkcd = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
    for j, n in enumerate(xkcd):
        xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
        xkcd_colors.append(xkcd)

    Ps = []
    sns.set_palette(sns.color_palette("Paired", 12))
    plt.figure(figsize=(20, 9))
    bottom = np.zeros(ntargets)
    for ii in range(ndim):
        ptemp = plt.bar(np.arange(1, 1 + ntargets), S[:, ii], bottom=bottom, width=.25)
        bottom = bottom + S[:, ii]  # reset bottom to new height
        Ps.append(ptemp)
    plt.ylabel('Sobol Scores')
    plt.ylim([0, 1.1])
    # plt.title('Sobol indices by pca mode')
    # plt.xticks(t, ('PCA1','PCA2','PCA3','PCA4','PCA5','PCA6'))
    # X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
    #            'sigk1', 'sigk2', 'sigw1', 'sigw2',
    #            'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']
    plt.legend(((p[0] for p in Ps)), (l for l in feature_labels),
           fancybox=True, shadow=True,
           loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ndim)

    # plot explained variance
    if extra_line_plot is not None:
        assert len(extra_line_plot) >= ntargets, "extra info must be the same length as the targets."
        plt.plot(np.arange(1, 1 + ntargets), extra_line_plot[:ntargets], '--ok')
    return plt

print("Performing grid search for multiple target fits one at a time...")
pce_param_grid = [
    {'order': [1, 2, 3],
     'mindex_type': ['total_order', 'hyperbolic', ],
     'fit_type': ['LassoCV', 'linear', 'ElasticNetCV']}
]
model = pypce.pcereg()

pceMCV = MultiOutputCV(regressor_model=model, regressor_param_grid=pce_param_grid,n_jobs=8)


# # test fit with one target
# pceMCV.fit(X,y)

# # test with multiple targets
# pceMCV.fit(X,Y)

# ipdb.set_trace()

################################################################
# PCA Target Transform
################################################################
print("Performing PCA transform...")
# test target transform
# Load the data
datadir = '/Users/kchowdh/Research/sparc_tests/'
X = np.load(datadir + 'data/X_' + 'heat_flux_wall' + '.npy')
Y = np.load(datadir + 'data/Y_' + 'heat_flux_wall' + '.npy')

# Scale target Y (use inverse transform for new data)
# Each row is a functional output, so we should scale each row
target_scaler = pypce.MinMaxTargetScaler(target_range=(0, 1))
Y_scaled = target_scaler.fit_transform(Y.T).T

# Lower and Upper Bound for features in X (for future scaling)
X_LB = np.array([.85, .85, .85, 0.7, 0.8, 0.3, 0.7, 0.0784, 0.38, 0.31, 1.19, 1.05])
X_UB = np.array([1.15, 1.15, 1.15, 1.0, 1.2, 0.7, 1.0, 0.1024, 0.42, 0.40, 1.31, 1.45])

X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
               'sigk1', 'sigk2', 'sigw1', 'sigw2',
               'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']

# Scale features X (use inverse transform for new data)
feature_scaler = pypce.DomainScaler(a=X_LB, b=X_UB, domain_range=(-1, 1))
X_scaled = feature_scaler.fit_transform(X)

pca = PCATargetTransform(n_components='auto', cutoff=1e-3)
pca.fit(Y_scaled)
Yhat_scaled = pca.transform(Y_scaled)

from sklearn.metrics import make_scorer
from collections import defaultdict
cv_test_scores = defaultdict(list)
cv_train_scores = defaultdict(list)
def custom_metric(y_true,y_pred,**kwargs):
    return np.linalg.norm(y_true - y_pred)/np.linalg.norm(y_true)
scorer = make_scorer(custom_metric,greater_is_better=False) 
# scorer = 'neg_root_mean_squared_error'
nprocs = 8
S_pce, S_rf, S_ab, S_gb = [], [], [], []
S_mlp, S_knn = [], []

import time as T
start = T.time()
for nn in range(Yhat_scaled.shape[1]-7):
    print("\nWorking on mode %i ...\n" %(nn+1))
    y = Yhat_scaled[:,nn]
    ################################################################
    # PCE Fit
    ################################################################
    pce_param_grid = [{'order': [1, 2, 3],
                       'mindex_type': ['total_order', 'hyperbolic', ],
                       'fit_type': ['LassoCV', 'linear', 'ElasticNetCV']}
                     ]
    model = pypce.pcereg()

    pceMCV = MultiOutputCV(regressor_model=model, regressor_param_grid=pce_param_grid,n_jobs=nprocs,score_metric=scorer)


    print("\n" + "*"*80 + "\nFitting with PCEs...\n" + "*"*80 + "\n")
    pceMCV.fit(X_scaled, y)
    S = pceMCV.feature_importance()
    S_pce.append(S)
    # plt = plot_feature_importance(S,X_col_names,pca.cumulative_error)
    print("Best PCE fit score: ", pceMCV.pceCV.best_score_)
    cv_test_scores['pce'].append(pceMCV.pceCV.best_score_)
    cv_score_argmin = pceMCV.pceCV.best_index_
    cv_train_scores['pce'].append(pceMCV.pceCV.cv_results_['mean_train_score'][cv_score_argmin])

    ################################################################
    # random forest regressor test
    ################################################################ 
    print("\n" + "*"*80 + "\nFitting with Random Forests...\n" + "*"*80 + "\n")
    from sklearn.ensemble import RandomForestRegressor

    # get cv from above
    cv = pceMCV.pceCV.cv

    # Create the random grid
    # 'min_samples_leaf': [2,.01,.05]
    # 'max_depth': [2,4,8,16] 
    # 'max_samples': [.5,.99],
    rf_param_grid = {'n_estimators': [10000,20000],
                     'max_features': ['log2'],
                     'max_depth': [4,5,6]
                     }
    # scorer = 'r2'
    rfreg = RandomForestRegressor(warm_start=False)
    rfregCV = GridSearchCV(rfreg,rf_param_grid,scoring=scorer,cv=cv,verbose=1,n_jobs=nprocs,return_train_score=True)

    rfregCV.fit(X_scaled,y)
    best_rf_estimator = rfregCV.best_estimator_

    S_rf.append(best_rf_estimator.feature_importances_)
    print("Best RF fit score: ", rfregCV.best_score_)
    cv_test_scores['rf'].append(rfregCV.best_score_)
    cv_score_argmin = rfregCV.best_index_
    cv_train_scores['rf'].append(rfregCV.cv_results_['mean_train_score'][cv_score_argmin])
    depths = [estimator.tree_.max_depth for estimator in best_rf_estimator.estimators_]

    ################################################################
    # adaboost
    ################################################################ 
    print("\n" + "*"*80 + "\nFitting with AdaBoost...\n" + "*"*80 + "\n")
    from sklearn.ensemble import AdaBoostRegressor

    # get cv from above
    cv = pceMCV.pceCV.cv
    scorer_ab = scorer
    # scorer = 'r2'

    ab_param_grid = {'n_estimators': [100,500],
                     'loss': ['linear','square']}
    ABreg = AdaBoostRegressor()
    ABregCV = GridSearchCV(ABreg,ab_param_grid,scoring=scorer,cv=cv,verbose=1,n_jobs=nprocs,return_train_score=True)

    ABregCV.fit(X_scaled,y)
    best_ab_estimator = ABregCV.best_estimator_

    S_ab.append(best_ab_estimator.feature_importances_)
    print("Best AdaBoost fit score: ", ABregCV.best_score_)
    cv_test_scores['ab'].append(ABregCV.best_score_)
    cv_score_argmin = ABregCV.best_index_
    cv_train_scores['ab'].append(ABregCV.cv_results_['mean_train_score'][cv_score_argmin])

    ################################################################
    # gradient boosting regressor
    ################################################################ 
    print("\n" + "*"*80 + "\nFitting with GradientBoost...\n" + "*"*80 + "\n")
    from sklearn.ensemble import GradientBoostingRegressor

    # get cv from above
    cv = pceMCV.pceCV.cv

    gb_param_grid = {'n_estimators': [500],
                     'learning_rate': [.1,.5],
                     'max_depth': [1,2],
                     'loss': ['ls','huber']
                     }
    GBreg = GradientBoostingRegressor()
    GBregCV = GridSearchCV(GBreg,gb_param_grid,scoring=scorer,cv=cv,verbose=1,n_jobs=nprocs,return_train_score=True)

    # X_transformed = pceMCV.pceCV.best_estimator_.fit_transform(X_scaled)

    GBregCV.fit(X_scaled,y)
    best_gb_estimator = GBregCV.best_estimator_

    S_gb.append(best_gb_estimator.feature_importances_)
    print("Best GradBoost fit score: ", GBregCV.best_score_)
    cv_test_scores['gb'].append(GBregCV.best_score_)
    cv_score_argmin = GBregCV.best_index_
    cv_train_scores['gb'].append(GBregCV.cv_results_['mean_train_score'][cv_score_argmin])

    ################################################################
    # MLP regressor
    ################################################################ 
    print("\n" + "*"*80 + "\nFitting with MLP...\n" + "*"*80 + "\n")
    from sklearn.neural_network import MLPRegressor

    # get cv from above
    cv = pceMCV.pceCV.cv

    mlp_param_grid = {'hidden_layer_sizes': [(1000)],
                     'solver': ['lbfgs','adam'],
                     'activation': ['relu'],
                     'max_iter': [1000],
                     'batch_size': ['auto'],
                     'learning_rate': ['constant','adaptive'],
                     'alpha': [.0001],
                     'tol': [1e-5]
                     }
    MLPreg = MLPRegressor()
    MLPregCV = GridSearchCV(MLPreg,mlp_param_grid,scoring=scorer,cv=cv,verbose=1,n_jobs=nprocs,return_train_score=True)

    X_transformed = pceMCV.pceCV.best_estimator_.fit_transform(X_scaled)
    # MLPregCV.fit(X_transformed,y) # exclude bias
    MLPregCV.fit(X_scaled,y)
    best_mlp_estimator = MLPregCV.best_estimator_

    # S_mlp.append(best_mlp_estimator.feature_importances_)
    print("Best MLP fit score: ", MLPregCV.best_score_)
    cv_test_scores['mlp'].append(MLPregCV.best_score_)
    cv_score_argmin = MLPregCV.best_index_
    cv_train_scores['mlp'].append(MLPregCV.cv_results_['mean_train_score'][cv_score_argmin])

    ################################################################
    # Nearest neighbor regressor
    ################################################################ 
    print("\n" + "*"*80 + "\nFitting with MLP...\n" + "*"*80 + "\n")
    from sklearn.neighbors import KNeighborsRegressor

    # get cv from above
    cv = pceMCV.pceCV.cv

    knn_param_grid = {'n_neighbors': [2,5,8,10],
                      'weights': ['uniform','distance'],
                      'algorithm': ['auto','brute'],
                      'leaf_size': [1,2,3,4],
                      'p': [2]
                     }
    kNNreg = KNeighborsRegressor()
    kNNregCV = GridSearchCV(kNNreg,knn_param_grid,scoring=scorer,cv=cv,verbose=1,n_jobs=nprocs,return_train_score=True)

    # X_transformed = pceMCV.pceCV.best_estimator_.fit_transform(X_scaled)
    # kNNregCV.fit(X_transformed,y) # exclude bias
    kNNregCV.fit(X_scaled,y)
    best_knn_estimator = kNNregCV.best_estimator_

    # S_knn.append(best_knn_estimator.feature_importances_)
    print("Best kNN fit score: ", kNNregCV.best_score_)
    cv_test_scores['knn'].append(kNNregCV.best_score_)
    cv_score_argmin = kNNregCV.best_index_
    cv_train_scores['knn'].append(kNNregCV.cv_results_['mean_train_score'][cv_score_argmin])
print("Total time is %.5f" %(T.time() - start))
################################################################
# plot validation scores
################################################################
import plotly.graph_objects as go

def plot_multiple_scatter(x,Y,Y_col_labels=None,ms=20,alpha=1,scale=1.0):
    # x can be a list of strings or a 1d array
    # Y can be a dictionary or a ndarray
    if isinstance(Y,np.ndarray):
        assert len(x) ==  Y.shape[1]
        assert len(Y_col_labels) == Y.shape[0]
        n_scatter = Y.shape[0]
    if isinstance(Y,dict):
        if Y_col_labels is None:
            Y_col_labels = list(Y.keys())
        n_scatter = len(Y_col_labels)
        Y = np.array([Y[k] for k in Y.keys()])
    fig = go.Figure()
    for i,y in enumerate(Y):
        fig.add_trace(go.Scatter(x=x, y=scale*y,
                                mode='lines+markers',
                                marker=dict(size=ms,opacity=alpha),
                                line=dict(width=2,dash='dash'),
                                name=Y_col_labels[i]))
    fig.update_layout(font=dict(size=18))
    return fig


x_labels = ["PCA mode %i" %(n+1) for n in range(len(cv_test_scores['pce']))]
col_labels_custom = ['PCE', 'RandForests', 'AdaBoost', 'GradBoost', 'MLP', 'kNN']
fig1 = plot_multiple_scatter(x_labels,cv_test_scores,Y_col_labels=col_labels_custom,alpha=.7,scale=-1.0)  
fig1.write_image("cv_test_scores.png",width=2000,height=1000)

cv_diff = np.array([np.array(cv_test_scores[k])-np.array(cv_train_scores[k]) for k in cv_train_scores.keys()])
col_labels = [k for k in cv_train_scores.keys()]
fig2 = plot_multiple_scatter(x_labels,cv_diff,Y_col_labels=col_labels_custom,alpha=.7,scale=-1.0)  
fig2.write_image("cv_diff_scores.png",width=2000,height=1000)

################################################################
# use plotly to plot interactive plots
################################################################
# print("Graphing...")
# # Need plotly, psutil, orca*, and pandas
# # * orca is a pain to install (easiest is to go to github and follow dmg installation guide)
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np


# class HiDFuncVisualizer():
#     def __init__(self, X, Y, feature_labels=None, target_labels=None, thin=None):
#         self.X = X
#         self.Y = Y
#         self.feature_labels = feature_labels
#         self.target_labels = target_labels
#         self.thin = thin

#     def _precompile(self):
#         # get dimensiona and shapes
#         self.nx, self.d = X.shape
#         if self.Y.ndim == 1:
#             self.Y = np.atleast_2d(self.Y).T
#         self.ny, self.ntargets = self.Y.shape
#         assert self.nx == self.ny, "X and Y have to have the same number of samples."
#         # check column names
#         if self.feature_labels is None:
#             self.feature_labels = ['x_s' + str(i + 1) for i in range(self.d)]
#         else:
#             assert len(self.feature_labels) == self.d, "feature labels need to be size " + self.d
#         if self.target_labels is None:
#             self.target_labels = ['y_' + str(i + 1) for i in range(self.ntargets)]
#         else:
#             assert len(self.target_labels) == self.ntargets, "target labels are not the right size. "
#         # size of plots
#         self.nrows = int(round(np.sqrt(self.d)))
#         self.ncols = round(self.d / self.nrows)

#     def compile(self):
#         # convert data to pandas dataframe
#         self._precompile()
#         if self.thin == None:
#             self.X_df = pd.DataFrame(data=self.X, columns=self.feature_labels)
#             self.Y_df = pd.DataFrame(data=self.Y, columns=self.target_labels)
#         elif isinstance(self.thin, int):
#             self.X_df = pd.DataFrame(data=self.X[::self.thin, :], columns=self.feature_labels)
#             self.Y_df = pd.DataFrame(data=self.Y[::self.thin, :], columns=self.target_labels)

#     def render(self, target_number=1, nrows=None, ncols=None, title=None):
#         # plot univariate, select which target_number you want to plot
#         if nrows is None and ncols is None:
#             nrows = self.nrows
#             ncols = self.ncols
#         assert nrows * ncols >= self.ntargets, "rows and cols must be bigger"
#         fig = make_subplots(rows=nrows, cols=ncols)
#         ycol = self.target_labels[target_number - 1]
#         for r in range(nrows):
#             for c in range(ncols):
#                 if r * ncols + c >= self.d:
#                     break
#                 else:
#                     xcol = self.feature_labels[r * ncols + c]
#                     fig.add_trace(
#                         go.Scatter(x=self.X_df[xcol],
#                                    y=self.Y_df[ycol],
#                                    mode='markers',
#                                    name=xcol + ' vs. ' + ycol),
#                         row=r + 1, col=c + 1
#                     )
#                 fig.update_xaxes(title_text=xcol, row=r + 1, col=c + 1)
#         fig.update_layout(title_text=ycol)
#         return fig

#     def render_corner(self, target_number=1, n_features=None):
#         # n_features = min(n_features,6)
#         if n_features is None:
#             self.mtx_nrows = len(self.feature_labels)
#             self.mtx_ncols = self.mtx_nrows
#         else:
#             self.mtx_nrows = len(self.feature_labels[:n_features])
#             self.mtx_ncols = self.mtx_nrows
#         if self.mtx_nrows > 6: warnings.warn("n_features > 6 does not render when using fig.show()")
#         # subplots
#         nrows, ncols = self.mtx_nrows, self.mtx_ncols
#         # spec = {'is_3d':True}
#         spec = {'type': 'Surface'}
#         specs = [[spec for j in range(ncols)] for i in range(nrows)]
#         # make diagonal univariate
#         for i in range(nrows):
#             specs[i][i] = {'type': 'Scatter'}
#         # fig = make_subplots(rows=nrows, cols=ncols,specs=specs)

#         fig = make_subplots(rows=nrows, cols=ncols, specs=specs, vertical_spacing=0.01, horizontal_spacing=.01,
#                             row_titles=self.feature_labels[:n], column_titles=self.feature_labels[:n_features])

#         zcol = self.target_labels[target_number - 1]
#         for r in range(1, nrows):
#             count = 0
#             for c in range(min(r, ncols)):
#                 # ix, iy = pairs[count]
#                 xcol1 = self.feature_labels[:n_features][r]
#                 xcol2 = self.feature_labels[:n_features][c]
#                 fig.add_trace(
#                     go.Scatter3d(x=self.X_df[xcol1],
#                                  y=self.X_df[xcol2],
#                                  z=self.Y_df[zcol],
#                                  name=xcol1 + ', ' + xcol2 + ' vs. ' + zcol,
#                                  mode='markers',
#                                  marker=dict(size=4),
#                                  opacity=.65),
#                     row=r + 1, col=c + 1
#                 )
#         for r in range(nrows):
#             c = r
#             xcol1 = self.feature_labels[:n_features][r]
#             xcol2 = self.feature_labels[:n_features][c]
#             fig.add_trace(
#                 go.Scatter(x=self.X_df[xcol1],
#                            y=self.Y_df[zcol],
#                            name=xcol1 + ' vs. ' + zcol,
#                            mode='markers',
#                            opacity=.65),
#                 row=r + 1, col=c + 1
#             )
#         fig.update_layout(title_text=zcol)
#         return fig


# # visualization
# # create a pandas data frame
# X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
#                'sigk1', 'sigk2', 'sigw1', 'sigw2',
#                'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']
# Yhat_col_names = ['PCA mode #' + str(i) + ' for heat flux' for i in range(1, 1 + Yhat_scaled.shape[1])]

# # convert data to pandas data frame
# X_scaled_df = pd.DataFrame(data=X_scaled, columns=X_col_names)
# Yhat_scaled_df = pd.DataFrame(data=Yhat_scaled, columns=Yhat_col_names)

# # use plotly to plot interactive plots
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# F = HiDFuncVisualizer(X_scaled, Yhat_scaled, feature_labels=X_col_names, target_labels=Yhat_col_names, thin=3)
# F.compile()
# print('saving figures...')
# fig1 = F.render(target_number=1)
# fig1.write_image("univariate.png",width=1200,height=1000)
# fig2 = F.render_corner(target_number=1)
# fig2.write_image("corner.png",width=1800,height=1600)
