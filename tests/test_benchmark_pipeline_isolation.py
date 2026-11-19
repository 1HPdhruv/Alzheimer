"""Stage 1 benchmark: fold-contained preprocessing (Test 3) and sampling isolation (Test 4). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from yoad.benchmark import pipeline as P
from yoad.benchmark.config import load_config
from yoad.benchmark.data import generate_synthetic
from yoad.benchmark.runner import run
from yoad.benchmark.splitting import group_partition


def _cfg(**over):
    base = {"experiment_id": "iso", "seed": 3, "dataset": {"params": {"n_participants": 160, "seed": 11}},
            "metrics": ["roc_auc"]}
    base.update(over)
    return load_config(base)


# ---------------------------------------------------------------- Test 3: fitting boundaries
def test_imputer_and_scaler_learn_only_from_training_rows():
    d = generate_synthetic(n_participants=200, missing_rate=0.2, seed=4)
    p = group_partition(d.y, d.groups, {"train": .7, "test": .3}, seed=0)
    X = d.X.copy()
    X.loc[p["test"], "syn_cog_1"] += 100.0                                # shift that only the test rows carry
    cfg = _cfg()
    pipe = P.build_pipeline(cfg, d.features, {"C": 1.0}, seed=0).fit(X.iloc[p["train"]], d.y.iloc[p["train"]])
    num = pipe.named_steps["columns"].named_transformers_["num"]
    col = [f.name for f in d.features if f.kind == "numeric"].index("syn_cog_1")
    train_median = X.iloc[p["train"]]["syn_cog_1"].median()
    global_median = X["syn_cog_1"].median()
    assert num.named_steps["impute"].statistics_[col] == pytest.approx(train_median)
    assert train_median != pytest.approx(global_median, abs=0.1)          # the assertion above can discriminate
    train_mean = X.iloc[p["train"]]["syn_cog_1"].fillna(train_median).mean()
    assert num.named_steps["scale"].mean_[col] == pytest.approx(train_mean)
    assert abs(train_mean - X["syn_cog_1"].fillna(global_median).mean()) > 10   # global scaling would differ grossly


def test_feature_selection_scores_come_from_training_rows_only():
    d = generate_synthetic(n_participants=200, missing_rate=0.0, seed=5)
    p = group_partition(d.y, d.groups, {"train": .7, "test": .3}, seed=0)
    cfg = _cfg(feature_selection={"method": "anova", "k": 3}, preprocessing={"scaling": "none"})
    pipe = P.build_pipeline(cfg, d.features, {"C": 1.0}, seed=0).fit(d.X.iloc[p["train"]], d.y.iloc[p["train"]])
    Xt = pipe.named_steps["columns"].transform(d.X.iloc[p["train"]])
    expected, _ = f_classif(Xt, d.y.iloc[p["train"]])
    np.testing.assert_allclose(pipe.named_steps["select"].scores_, expected)


class FitSpy(BaseEstimator, TransformerMixin):
    """Records the row index of every DataFrame it is fitted on (test-only hook)."""
    fitted_on: list = []

    def fit(self, X, y=None):
        FitSpy.fitted_on.append(set(X.index))
        return self

    def transform(self, X):
        return X


def test_runner_never_fits_preprocessing_on_validation_or_evaluation_rows(tmp_path, monkeypatch):
    """Regression test: fails if any learned step were fitted before splitting or on held-out rows."""
    monkeypatch.setitem(P.STEP_HOOKS, "fit_spy", FitSpy)
    FitSpy.fitted_on = []
    cfg = {"experiment_id": "spy", "seed": 3, "dataset": {"params": {"n_participants": 160, "seed": 11}},
           "preprocessing": {"extra_steps": ["fit_spy"]}, "model": {"search_space": {"C": [0.1, 1.0]}},
           "validation": {"validation_fraction": 0.2}, "threshold": {"method": "youden"}, "metrics": ["roc_auc"]}
    r = run(cfg, results_base=tmp_path, run_id="r")
    assert r["status"] == "success", r["error"]
    # reconstruct the partitions exactly as the runner made them
    d = generate_synthetic(n_participants=160, seed=11)
    y, g = d.y.to_numpy(), d.groups.to_numpy()
    outer = group_partition(y, g, {"train": .75, "test": .25}, 3, True, ("test",))
    vf = 0.2 / 0.75
    inner = group_partition(y[outer["train"]], g[outer["train"]], {"fit": 1 - vf, "validation": vf}, 3 + 100, True, ("validation",))
    fit_rows = set(outer["train"][inner["fit"]])
    held_out = set(outer["test"]) | set(outer["train"][inner["validation"]])
    assert len(FitSpy.fitted_on) == 2 * 3 + 1                             # 2 candidates x 3 inner folds + final refit
    for rows in FitSpy.fitted_on:
        assert rows <= fit_rows and not rows & held_out
    assert FitSpy.fitted_on[-1] == fit_rows


def test_global_fit_would_be_caught():
    """Shows the spy assertion discriminates: a pipeline fitted on all rows touches held-out rows."""
    FitSpy.fitted_on = []
    d = generate_synthetic(n_participants=60, seed=1)
    FitSpy().fit(d.X)
    p = group_partition(d.y, d.groups, {"train": .75, "test": .25}, seed=0)
    assert FitSpy.fitted_on[0] & set(p["test"])


# ---------------------------------------------------------------- Test 4: sampling isolation
class CountingSampler(P.RandomOverSampler):
    calls: list = []

    def fit_resample(self, X, y):
        CountingSampler.calls.append(len(y))
        return super().fit_resample(X, y)


def test_resampler_only_touches_fit_rows_and_never_prediction_rows():
    CountingSampler.calls = []
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(100, 3)); ytr = (rng.random(100) < 0.2).astype(int)
    Xte = rng.normal(size=(37, 3)); Xte_copy = Xte.copy()
    clf = P.ResampledClassifier(LogisticRegression(), CountingSampler(seed=0)).fit(Xtr, ytr)
    assert CountingSampler.calls == [100]
    assert clf.resampled_class_counts_[0] == clf.resampled_class_counts_[1] == (ytr == 0).sum()
    p = clf.predict_proba(Xte)
    assert p.shape == (37, 2) and CountingSampler.calls == [100]          # prediction did not resample
    np.testing.assert_array_equal(Xte, Xte_copy)                          # evaluation data unchanged


def test_undersampler_balances_without_replacement():
    y = np.array([0] * 30 + [1] * 10); X = np.arange(40).reshape(-1, 1)
    Xr, yr = P.RandomUnderSampler(seed=1).fit_resample(X, y)
    assert (yr == 0).sum() == (yr == 1).sum() == 10 and len(set(Xr.ravel())) == 20


def test_runner_resamples_only_the_fitting_partition(tmp_path):
    cfg = {"experiment_id": "samp", "seed": 3, "dataset": {"params": {"n_participants": 160, "positive_fraction": 0.2, "seed": 11}},
           "sampling": {"method": "random_oversample"}, "validation": {"validation_fraction": 0.2},
           "threshold": {"method": "youden"}, "metrics": ["roc_auc", "brier"]}
    r = run(cfg, results_base=tmp_path, run_id="r")
    f = r["folds"][0]
    assert r["status"] == "success", r["error"]
    assert f["resampling"]["rows_into_sampler"] == f["partitions"]["fit"]["n_observations"]
    assert f["resampling"]["rows_after_sampler"] > f["resampling"]["rows_into_sampler"]
    assert f["partitions"]["evaluation"]["n_observations"] + f["partitions"]["validation"]["n_observations"] \
        + f["partitions"]["fit"]["n_observations"] == len(generate_synthetic(n_participants=160, positive_fraction=0.2, seed=11).y)


def test_planned_sampler_fails_clearly():
    with pytest.raises(NotImplementedError, match="smotenc"):
        P.build_pipeline({**_cfg(), "sampling": {"method": "smotenc"}}, generate_synthetic(seed=0).features, {}, 0)
