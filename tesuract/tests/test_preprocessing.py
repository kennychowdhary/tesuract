import tesuract
import unittest
import numpy as np
import pytest
from tesuract.preprocessing import *
from sklearn.datasets import make_low_rank_matrix

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)


@pytest.mark.unit
class Test_Target_Scaler(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rn = np.random.RandomState(399123)

    def test_target_scaler(self):
        Y = 3.5 * self.rn.rand(10, 100) + 2
        scaler = MinMaxTargetScaler()
        scaler.fit(Y)
        Yhat = scaler.fit_transform(Y)
        assert (
            np.amin(Yhat) == 0.0 and np.amax(Yhat) == 1.0
        ), "Scaler is not working properly"

    def test_target_scaler_inverse(self):
        Y = 3.5 * self.rn.rand(10, 100) + 2
        scaler = MinMaxTargetScaler()
        scaler.fit(Y)
        Yhat = scaler.fit_transform(Y)
        assert (
            np.sum(scaler.inverse_transform(Yhat) - Y) <= 1e-10
        ), "Inverse transform not working properly."

    def test_minmaxtargetscaler(self):
        Y = self.rn.rand(10, 3)
        ts = MinMaxTargetScaler(target_range=(1.0, 2.0))
        ts.fit(Y)
        Yt = ts.transform(Y)
        assert np.amin(Yt) >= 1.0, "transform out of range"
        assert np.amax(Yt) <= 2.0, "transform out of range"
        error = np.linalg.norm(ts.inverse_transform(Yt) - Y)
        assert error <= 1e-15, "transform and inverse not working"

    def test_simple_mapping(self):
        Y = np.array([np.zeros(3), np.ones(3)])
        ts = MinMaxTargetScaler(target_range=(3.0, 10.0))
        ts.fit(Y)
        Yt = ts.transform(Y)
        assert np.mean(Yt[0]) == 3.0, "transform out of range"
        assert np.mean(Yt[1]) == 10.0, "transform out of range"

    def test_simple_mapping2(self):
        Y = np.array([3 + np.zeros(3), 5 * np.ones(3)])
        ts = MinMaxTargetScaler(target_range=(0.0, 1.0))
        ts.fit(Y)
        Yt = ts.transform(Y)
        assert np.mean(Yt[0]) == 0.0, "transform out of range"
        assert np.mean(Yt[1]) == 1.0, "transform out of range"


@pytest.mark.unit
class Test_Domain_Scaler(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rn = np.random.RandomState(399123)

    def test_domain_scaler(self):
        ds = DomainScaler(dim=3, input_range=[(1.0, 3.0), (1.0, 3.0), (1.0, 3.0)])
        Y = np.array([1 + np.zeros(3), 3 * np.ones(3)])
        Yt = ds.fit_transform(Y)
        assert np.mean(Yt[0]) == -1.0, "transform is broken"
        assert np.mean(Yt[1]) == 1.0, "transform is broken"
        error = np.linalg.norm(ds.inverse_transform(Yt) - Y)
        assert error <= 1e-15, "inverse transform not working"

    def test_nonuniform_input_range(self):
        ds = DomainScaler(dim=3, input_range=[(1.0, 3.0), (1.0, 4.0), (1.0, 5.0)])
        Y = np.array([[1, 1, 1], [3.0, 4.0, 5.0]])
        Yt = ds.fit_transform(Y)
        assert np.mean(Yt[0]) == -1.0, "transform is broken"
        assert np.mean(Yt[1]) == 1.0, "transform is broken"
        error = np.linalg.norm(ds.inverse_transform(Yt) - Y)
        assert error <= 1e-15, "inverse transform not working"

    def test_nonuniform_input_range2(self):
        ds = DomainScaler(dim=3, input_range=[(1.0, 3.0), (1.1, 4.0), (1.2, 5.0)])
        Y = np.array([[1, 1.1, 1.2], [3.0, 4.0, 5.0]])
        Yt = ds.fit_transform(Y)
        assert np.mean(Yt[0]) == -1.0, "transform is broken"
        assert np.mean(Yt[1]) == 1.0, "transform is broken"
        error = np.linalg.norm(ds.inverse_transform(Yt) - Y)
        assert error <= 1e-15, "inverse transform not working"

    def test_nondefault_input_output_range(self):
        ds = DomainScaler(
            dim=3,
            input_range=[(1.0, 3.0), (1.1, 4.0), (1.2, 5.0)],
            output_range=(10, 11.0),
        )
        Y = np.array([[1, 1.1, 1.2], [3.0, 4.0, 5.0]])
        Yt = ds.fit_transform(Y)
        assert np.mean(Yt[0]) == 10.0, "transform is broken"
        assert np.mean(Yt[1]) == 11.0, "transform is broken"
        error = np.linalg.norm(ds.inverse_transform(Yt) - Y)
        assert error <= 1e-15, "inverse transform not working"


@pytest.mark.unit
class Test_Log_Scaler(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rn = np.random.RandomState(399123)
        self.Y = self.rn.rand(100, 5)
        self.LT = LogTransform()

    def test_log_transform(self):
        Y = self.Y
        logY = np.log(Y)
        self.LT.fit(Y)
        assert (
            np.linalg.norm(self.LT.transform(Y) - logY) < 1e-15
        ), ".transform method is not working. "

    def test_log_inverse_transform(self):
        Y = self.Y
        logY = np.log(Y)
        self.LT.fit(Y)
        Yhat = self.LT.transform(Y)
        assert (
            np.linalg.norm(self.LT.inverse_transform(Yhat) - Y) < 1e-15
        ), ".transform method is not working. "

    def test_log_transform_neg_check(self):
        Y = self.rn.randn(100, 5)
        with pytest.raises(AssertionError):
            self.LT.transform(Y)


@pytest.mark.unit
class Test_Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rn = np.random.RandomState(399123)
        self.Y = self.rn.rand(10, 5)
        self.LT = LogTransform()

    def test_gen_pipeline_log(self):
        ncomp = 4
        pipe = gen_scale_pipeline(log=True, scale=False, pca=False, n_components=ncomp)
        Y = self.Y
        pipe.fit(Y)
        Yhat = pipe.transform(Y)
        Yr = pipe.inverse_transform(Yhat)
        # assert Yhat.shape[1] == ncomp, "new dimensions do not match n_components. Check PCA."
        assert Yr.shape == Y.shape, "original and reconstructed shapes do no match. "
        assert np.mean(np.mean((Yr - Y) ** 2, axis=1)) < 1e-16

    def test_gen_pipeline_log_scale(self):
        ncomp = 4
        pipe = gen_scale_pipeline(log=True, scale=True, pca=False, n_components=ncomp)
        Y = self.Y
        pipe.fit(Y)
        Yhat = pipe.transform(Y)
        Yr = pipe.inverse_transform(Yhat)
        # assert Yhat.shape[1] == ncomp, "new dimensions do not match n_components. Check PCA."
        assert Yr.shape == Y.shape, "original and reconstructed shapes do no match. "
        assert np.mean(np.mean((Yr - Y) ** 2, axis=1)) < 1e-16

    def test_gen_pipeline_log_scale_pca(self):
        ncomp = self.Y.shape[1]
        pipe = gen_scale_pipeline(
            log=True, scale=True, pca=True, n_components=ncomp, svd_solver="auto"
        )
        Y = self.Y
        pipe.fit(Y)
        Yhat = pipe.transform(Y)
        Yr = pipe.inverse_transform(Yhat)
        assert (
            Yhat.shape[1] == ncomp
        ), "new dimensions do not match n_components. Check PCA."
        assert Yr.shape == Y.shape, "original and reconstructed shapes do no match. "
        rmse = np.mean((Yr - Y) ** 2, axis=1) / np.mean(Y**2, axis=1)
        assert (
            rmse.mean() < 1e-16
        ), "PCA should be exact, but isn't. Something is wrong."


@pytest.mark.unit
class Test_PCA_Target_Transform(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rn = np.random.RandomState(399123)
        self.K = 8
        self.Y = make_low_rank_matrix(
            n_samples=100,
            n_features=75,
            effective_rank=self.K,
            tail_strength=0.01,
            random_state=self.rn,
        )
        U, S, VT = np.linalg.svd(self.Y, full_matrices=False)
        S[: self.K] = 0.0
        self.Y = np.dot(np.dot(U, np.diag(S)), VT)

    def test_pca_auto(self):
        Y = self.Y
        # pca1 = PCA(n_components=50) # sklearn's PCA
        cutoff = 1e-2
        pca = PCATargetTransform(n_components="auto", exp_var_cutoff=1 - cutoff)
        pca.fit(Y)
        assert 1 - pca.cumulative_error[-1] <= cutoff, "cutoff not working. "
        assert pca.K <= self.K + 2, "auto cutoff isn't working to desired amount."

    def test_pca_cumulative_error(self):
        Y = self.Y
        pca1 = PCA(n_components=50)  # sklearn's PCA
        pca2 = PCATargetTransform(n_components=20)
        pca1.fit(Y)
        pca2.fit(Y)
        ce1 = np.cumsum(pca1.explained_variance_ratio_)[: pca2.K]
        ce2 = pca2.cumulative_error
        assert np.linalg.norm(ce1 - ce2) / np.linalg.norm(ce1) <= 1e-12
