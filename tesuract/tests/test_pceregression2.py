import tesuract
import unittest
import numpy as np
import warnings, pdb
import time as T
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


@pytest.mark.regression
class TestPCERegression(unittest.TestCase):
    def test_checking_predict_does_recompute_mindex(self):
        # this test will ensure the model selector selector works
        X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
        p = tesuract.PCEReg(order=2)
        p.fit(X, y)
        p.predict(X)
        Xnew = X[:, :-2]
        p.fit(Xnew, y)
        p.predict(Xnew)
        print("# times computing mindex:", p._mindex_compute_count_)
        assert (
            p._mindex_compute_count_ == 2
        ), "mindex should be recomputed when X dim changes. "

    def test_checking_predict_does_not_recompute_mindex(self):
        # this is a test to count the number of times mindex is computed
        X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
        p = tesuract.PCEReg(order=2)
        p.fit(X, y)
        p.predict(X)
        Xnew, ynew = X[10:], y[10:]
        p.fit(Xnew, ynew)
        p.predict(Xnew)
        print("# times mindex is computed: ", p._mindex_compute_count_)
        assert (
            p._mindex_compute_count_ == 1
        ), "mindex should NOT be recomputed when X row size changes."

    def test_k_fold_mindex_count(self):
        # this is a test to count the number of times mindex is computed
        X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
        X = 2 * X - 1  # scale to [-1,1]
        kf = KFold(n_splits=5)
        p = tesuract.PCEReg()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p.fit(X_train, y_train)
            score = np.sum(1 - p.predict(X_test) ** 2 / y_test**2)
            # print(score)
        print("# times mindex is computed: ", p._mindex_compute_count_)
        assert (
            p._mindex_compute_count_ == 1
        ), "mindex should NOT be recomputed when X row size changes."

    def test_k_fold_mindex_count(self):
        # this is a test to count the number of times mindex is computed
        X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
        X = 2 * X - 1  # scale to [-1,1]
        params = {"order": [1, 2, 3]}
        estimator = tesuract.PCEReg(fit_type="LassoCV")
        grid = GridSearchCV(estimator, params)
        grid.fit(X, y)
        # print(grid.cv_results_)
        print("# times computed...", grid.best_estimator_._mindex_compute_count_)
        assert (
            grid.best_estimator_._mindex_compute_count_ == 1
        ), "Best estimator should only compute mindex once!"

    def test_feature_importance_friedman_example(self):
        # this is a test to count the number of times mindex is computed
        X, y = make_friedman1(n_samples=100, n_features=10, random_state=0)
        X = 2 * X - 1  # scale to [-1,1]
        estimator = tesuract.PCEReg(order=4, fit_type="LassoCV")
        estimator.fit(X, y)
        fi = estimator.feature_importances_
        print(fi > 0.05)
        assert (
            np.sum(fi > 0.05) == 5
        ), "friedman function should have 5 important features only."

    def test_grid_search_cv_picks_right_order(self):
        #
        rn = np.random.RandomState(99)
        X = 2 * rn.rand(50, 5) - 1
        fi_index = 3
        x = X[:, 3]
        # y = (1./8)*(35.*x**4 - 30*x**4 + 3)
        y = (1.0 / 2) * (5 * x**3 - 3 * x)
        params = {"order": [1, 2, 3, 4]}
        estimator = tesuract.PCEReg(fit_type="LassoCV")
        start = T.time()
        grid = GridSearchCV(estimator, params, cv=KFold(n_splits=5))
        grid.fit(X, y)
        end = T.time()
        print("Total grid search time is {0:.2f} seconds.".format(end - start))
        # print(grid.cv_results_)
        print("\nbest parameters:", grid.best_params_)
        fi = grid.best_estimator_.feature_importances_
        # print("\nfeatures:", grid.best_estimator_.feature_importances_)
        assert (
            grid.best_params_["order"] == 3
        ), "Grid search should find the best order 4, but it is not. Something is wrong."
        assert (
            np.argmax(fi) == fi_index
        ), "Not finding the right feature importance index."

    def test_grid_search_cv_picks_right_order_and_features(self):
        #
        rn = np.random.RandomState(99)
        X = 2 * rn.rand(50, 5) - 1
        fi_index = 1
        x = X[:, fi_index]
        y = (1.0 / 8) * (35.0 * x**4 - 30 * x**4 + 3)
        # y = (1./2)*(5*x**3 - 3*x)
        params = {"order": [1, 2, 3, 4]}
        estimator = tesuract.PCEReg(fit_type="LassoCV")
        start = T.time()
        grid = GridSearchCV(estimator, params, cv=KFold(n_splits=5))
        grid.fit(X, y)
        end = T.time()
        print("Total grid search time is {0:.2f} seconds.".format(end - start))
        print(grid.cv_results_)
        print("\nbest parameters:", grid.best_params_)
        fi = grid.best_estimator_.feature_importances_
        # print("\nfeatures:", fi)
        assert (
            grid.best_params_["order"] == 3
        ), "Grid search should find the best order 4, but it is not. Something is wrong."
        assert (
            np.argmax(fi) == fi_index
        ), "Not finding the right feature importance index."
        # print("# times computed...", grid.best_estimator_._mindex_compute_count_)


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# regression test for multi output pca regressor
@pytest.mark.regression
class TestMRegressionWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = np.loadtxt(relpath + "/tests/data/rom_test_X.txt")
        self.Y = np.loadtxt(relpath + "/tests/data/rom_test_Y.txt")
        X, Y = self.X, self.Y

        from sklearn.model_selection import KFold

        self.kf = KFold(n_splits=2)

    def test_simplified_model_fit_with_single_param_for_each_comp(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False,
            scale=False,
            pca=True,
            n_components="auto",
            whiten=True,
            exp_var_cutoff=0.5,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        start = T.time()
        cvscore = cross_val_score(regmodel, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

        n_components = len(regmodel.best_params_)
        reg_custom_list = ["pce" for i in range(n_components)]
        reg_param_list = regmodel.best_params_
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False, scale=False, pca=True, n_components=n_components, whiten=True
        )
        regmodel_opt = tesuract.MRegressionWrapperCV(
            regressor=reg_custom_list,
            reg_params=reg_param_list,
            custom_params=True,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        start = T.time()
        cvscore = cross_val_score(regmodel_opt, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

    def test_simplified_model_fit_with_uninstantiated_PCA_target_transform(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        # target_transform = tesuract.preprocessing.target_pipeline_custom(log=False,scale=False,pca=True,n_components='auto',whiten=True,cutoff=.5)
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=PCA,
            target_transform_params={"n_components": 4},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel._fit_target_transform(Y)

    def test_target_transform_has_certain_attributes_without_target_instantiation(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        # FactorAnalysis which has no inverse transform
        from sklearn.decomposition import FactorAnalysis

        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=FactorAnalysis,
            target_transform_params={"n_components": 8},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        with self.assertRaises(Exception):
            regmodel._fit_target_transform(Y)

    def test_target_transform_has_certain_attributes_w_target_instantiation(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        # FactorAnalysis which has no inverse transform
        from sklearn.decomposition import FactorAnalysis

        FA = FactorAnalysis(n_components=4)
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=FA,
            target_transform_params=None,
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        with self.assertRaises(Exception):
            regmodel._fit_target_transform(Y)

    def test_simplified_model_fit_with_instantiated_PCA_target_transform(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        # target_transform = tesuract.preprocessing.target_pipeline_custom(log=False,scale=False,pca=True,n_components='auto',whiten=True,cutoff=.5)
        target_transform = PCA(n_components=2)
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        start = T.time()
        cvscore = cross_val_score(regmodel, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

        n_components = len(regmodel.best_params_)
        reg_custom_list = ["pce" for i in range(n_components)]
        reg_param_list = regmodel.best_params_
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False, scale=False, pca=True, n_components=n_components, whiten=True
        )
        regmodel_opt = tesuract.MRegressionWrapperCV(
            regressor=reg_custom_list,
            reg_params=reg_param_list,
            custom_params=True,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        start = T.time()
        cvscore = cross_val_score(regmodel_opt, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

    def test_multi_target_init_with_custom_param_list(self):
        # uses the best params as the new set of param grid space
        # not efficient but can be faster than original space
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False,
            scale=False,
            pca=True,
            n_components="auto",
            whiten=True,
            exp_var_cutoff=0.5,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        start = T.time()
        cvscore = cross_val_score(regmodel, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

        new_params = regmodel.best_params_
        new_regressors = ["pce" for i in range(len(new_params))]
        regmodel_opt = tesuract.MRegressionWrapperCV(
            regressor=new_regressors,
            reg_params=new_params,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        start = T.time()
        cvscore = cross_val_score(regmodel_opt, X, Y, cv=5, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        print("total time is ", T.time() - start)

    def test_multi_target_init_with_custom_param_list2(self):
        # uses the best params as the new set of param grid space
        # not efficient but can be faster than original space
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = "pce"  # will be a list automaticall if string
        param_list = pce_grid

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.PCATargetTransform(
            n_components="auto",
            whiten=True,
            exp_var_cutoff=0.5,
        )
        print(target_transform.fit_transform(Y).shape)
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        print("custom params:", regmodel.custom_params)  # should be
        regmodel.fit(X, Y)

    def test_rom_w_single_regressor_as_list(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False, scale=False, pca=True, n_components=4, whiten=True
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        cvscore = cross_val_score(regmodel, X, Y, cv=2, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))

    def test_rom_w_single_regressor_as_list_wo_pca(self):
        X, Y = self.X, self.Y[:, ::75]  # shorten output
        pce_grid = [
            {
                "order": list(range(2)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = ["pce"]
        param_list = [pce_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        # target_transform = tesuract.preprocessing.target_pipeline_custom(log=False,scale=False,pca=False,n_components=4,whiten=True)
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=None,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        cvscore = cross_val_score(regmodel, X, Y, cv=2, scoring="r2", n_jobs=1)
        print("cv r2 score: {0}%".format(-100 * np.round(cvscore.mean(), 4)))
        # get feature importances and weighted versions
        fi = regmodel.feature_importances_
        ws = regmodel.sobol_weighted_
        assert ws.ndim == 1, "weighted sobol must be a single vector"
        # get explained variance ratio
        evr = regmodel.explained_variance_ratio_
        assert np.amin(evr) == np.amax(
            evr
        ), "The explained variance ratio when target transform = None should be equal."
        # print("explained var ratio when target transform is None: ", evr)

    def test_rom_w_single_regressor_as_str(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        regressors = "pce"
        param_list = pce_grid

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False,
            scale=False,
            pca=True,
            n_components=3,
            whiten=True,
            exp_var_cutoff=1 - 0.25,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
        # test feature importance and explained var ratio for pca
        fi = regmodel.feature_importances_
        ws = regmodel.sobol_weighted_
        assert ws.ndim == 1, "weighted sobol must be a single vector"
        # get explained variance ratio
        evr = regmodel.explained_variance_ratio_
        print("explained variance ratio when using pca:", evr)

    def test_rom_w_multiple_regressors(self):
        X, Y = self.X, self.Y
        pce_grid = [
            {
                "order": list(range(1, 3)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]
        # random forest fit
        rf_grid = {
            "n_estimators": [50, 100],
            "max_features": ["auto"],
            "max_depth": [5, 10],
        }
        regressors = ["pce", "rf"]
        param_list = [pce_grid, rf_grid]

        def my_scorer(ytrue, ypred):
            mse = np.mean((ytrue - ypred) ** 2) / np.mean(ytrue**2)
            return -mse

        custom_scorer = make_scorer(my_scorer, greater_is_better=True)
        target_transform = tesuract.preprocessing.target_pipeline_custom(
            log=False,
            scale=False,
            pca=True,
            n_components="auto",
            whiten=True,
            exp_var_cutoff=1 - 0.25,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=regressors,
            reg_params=param_list,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=1,
            scorer=custom_scorer,
            verbose=0,
        )
        regmodel.fit(X, Y)
