"""Synthetic unit tests: registry, metrics, selection rule, source mappers and estimators (no real data anywhere)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from yoad.experiments.today_audit.audit_core import cal_intercept as frozen_intercept, cal_slope as frozen_slope
from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.multimodal import metrics as MT, models as M, registry as R, selection as S, sources as SRC
from tests.multimodal import synthetic_cohort as syn


# ------------------------------------------------------------------------------------------------ registry
def test_reference_sets_equal_the_frozen_sets_and_matrix_is_well_formed():
    assert R.resolve("FS08_CORE7") == frozen.PREDICTOR_SETS["primary"]
    assert R.resolve("FS08a_DEM_APOE") == frozen.PREDICTOR_SETS["S-noDx"]
    assert R.resolve("FS08b_TRANSPORT") == ["age", "female", "education_years", "apoe4", "mmse", "faq"]
    assert "adas13" not in R.resolve("FS08b_TRANSPORT")
    assert R.resolve("FS19_CORE7_MRIFULL", ["ST1SV", "ST2CV"])[-2:] == ["fs_ST1SV", "fs_ST2CV"]
    m = R.experiment_matrix(set(R.POPULATIONS))
    ids = [c["id"] for c in m]
    assert len(ids) == len(set(ids))
    assert all(c["population"] in R.POPULATIONS and c["model"] in R.MODELS for c in m)
    assert not [c for c in m if c["model"] == "cox_l2" and c["population"] != "P_ALL"]
    plan = R.contrast_plan(ids)
    assert plan and all(p["candidate"] in ids and p["comparator"] in ids for p in plan)
    assert not [c for c in R.experiment_matrix({"P_ALL"}) if c["population"] != "P_ALL"]     # gated populations drop out


def test_grids_are_bounded_deterministic_and_hash_tracks_the_plan(monkeypatch):
    for mid, spec in R.MODELS.items():
        if "grid" in spec:
            g1, g2 = R.grid_configs(mid), R.grid_configs(mid)
            assert g1 == g2 and 1 <= len(g1) <= spec["n_iter"]
            assert all(set(c) == set(spec["grid"]) for c in g1)
    h = R.config_sha256(["ST1SV"])
    assert h == R.config_sha256(["ST1SV"]) and h != R.config_sha256(["ST2SV"])
    monkeypatch.setattr(R, "OUTER_REPEATS", 4)
    assert R.config_sha256(["ST1SV"]) != h


# ------------------------------------------------------------------------------------------------ metrics
def test_fast_metrics_equal_the_frozen_implementations():
    rng = np.random.default_rng(3)
    for _ in range(5):
        y = rng.binomial(1, 0.3, 300); p = np.clip(rng.beta(2, 5, 300) + 0.2 * y, 0.001, 0.999)
        p[:20] = np.round(p[:20], 1)                            # ties
        m = MT.point_metrics(y, p)
        assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
        assert m["average_precision"] == pytest.approx(average_precision_score(y, p), abs=1e-12)
        assert m["calibration_intercept"] == pytest.approx(frozen_intercept(y, p), abs=1e-5)
        # the frozen slope is LogisticRegression(C=1e8) stopped at lbfgs' default tolerance; the Newton value is the exact MLE
        assert m["calibration_slope"] == pytest.approx(frozen_slope(y, p), abs=2e-3)
        X = np.column_stack([np.ones(len(y)), MT._logit(p)])
        mle = minimize(lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b)), [0.0, 1.0],
                       jac=lambda b: X.T @ (1 / (1 + np.exp(-(X @ b))) - y), method="BFGS", options={"gtol": 1e-10}).x
        assert m["calibration_slope"] == pytest.approx(mle[1], abs=1e-6)
        assert m["log_oe"] == pytest.approx(math.log(y.sum() / p.sum()))


def test_bootstrap_contrasts_and_suppressed_bins():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.3, 200); P = rng.random((2, 200))
    same = MT.paired_bootstrap(y, P, P.copy(), B=50)
    assert same["brier"]["difference"] == 0 and same["brier"]["ci_low"] == 0 and same["brier"]["ci_high"] == 0
    Pn = P.copy(); Pn[:, :10] = np.nan
    assert MT.paired_bootstrap(y, Pn, P, B=20)["n"] == 190     # only participants predicted by both
    lo, hi = MT.bootstrap(y, P, B=50)["roc_auc"]
    assert lo <= MT.repeat_summary(y, P)["roc_auc"][0] <= hi
    b = MT.calibration_bins(np.r_[np.ones(5), np.zeros(95)].astype(int), np.linspace(0.01, 0.99, 100))
    assert b["observed_proportion"].isna().sum() + (b["observed_proportion"] == 0).sum() == len(b)
    assert MT.classify(-0.01, 0.01, 0.02) == "inside margin" and MT.classify(0.03, 0.05, 0.02) == "outside margin"
    assert MT.classify(-0.03, 0.01, 0.02) == "inconclusive"


# ------------------------------------------------------------------------------------------------ selection
def _rec(i, model, feats, brier, auc=0.85, slope=1.0, oe=0.0):
    return {"id": i, "model": model, "features": feats, "brier": brier, "roc_auc": auc, "calibration_slope": slope, "log_oe": oe,
            "complete": True}


def test_selection_prefers_the_simplest_comparable_calibrated_candidate():
    core = list(R.resolve("FS08_CORE7"))
    pool = [_rec("best", "xgb_tuned", core + ["cdrsb"], 0.120), _rec("simple", "lr_l2_ref", core, 0.124),
            _rec("worse", "lr_l2_ref", ["age"], 0.180)]
    vs = {"simple": (0.004, 0.006, -0.01), "worse": (0.06, 0.01, -0.2)}
    out = S.select(pool, lambda b, c: vs[c])
    assert out["winner"] == "simple" and out["best_brier"] == "best"
    vs["simple"] = (0.004, 0.006, -0.03)                      # AUROC loss beyond the frozen 0.02 margin -> not comparable
    assert S.select(pool, lambda b, c: vs[c])["winner"] == "best"
    pool[1]["calibration_slope"] = 0.5; vs["simple"] = (0.004, 0.006, 0.0)
    assert S.select(pool, lambda b, c: vs[c])["winner"] == "best"
    for p in pool:
        p["log_oe"] = 0.5
    out = S.select(pool, lambda b, c: vs[c])
    assert out["winner"] == "best" and "flagged" in out["reason"]
    pool[0]["complete"] = False
    assert S.select(pool, lambda b, c: vs.get(c, (0, 1, 0)))["best_brier"] == "simple"


# ------------------------------------------------------------------------------------------------ sources
def test_long_measurements_map_ravlt_cdr_and_negative_codes():
    nb = pd.DataFrame({"RID": [1, 2], "VISDATE": ["2020-01-01", "2020-01-02"], "AVTOT1": [5, -4], "AVTOT2": [6, 7],
                       "AVTOT3": [7, 8], "AVTOT4": [8, 9], "AVTOT5": [10, 10], "AVDEL30MIN": [4, 2], "LDELTOTAL": [3, -1],
                       "TRABSCOR": [120, 300], "CATANIMSC": [15, 12]})
    cdr = pd.DataFrame({"RID": [1], "VISDATE": ["2020-01-01"], "CDRSB": [1.5], "CDGLOBAL": [0.5]})
    m, missing = SRC.long_measurements({"neurobat": nb, "cdr": cdr})
    v = m.set_index(["participant_id", "variable"])["value"]
    assert v[(1, "ravlt_immediate")] == 36 and np.isnan(v[(2, "ravlt_immediate")])        # one invalid trial -> missing
    assert v[(1, "ravlt_perc_forgetting")] == pytest.approx(60.0)
    assert np.isnan(v[(2, "ldeltotal")]) and v[(1, "cdrsb")] == 1.5
    assert "adas11" in missing                                                              # ADAS table not supplied


def test_mri_mapping_qc_duplicates_verification_and_plausibility():
    d = syn.make(60, seed=2)
    ok, info = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, SRC.verify_mri_codes(syn.datadic()))
    assert set(info["feature_status"].values()) == {"verified"}
    assert info["rows_qc_fail"] > 0 and not ok.duplicated(["participant_id", "date"]).any()
    assert 0.003 < ok["mri_hippocampus"].median() < 0.007                                   # ICV-normalised
    bad, info2 = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, SRC.verify_mri_codes(syn.datadic(contradict=("ST29SV",))))
    assert "mri_hippocampus" not in bad.columns and info2["feature_status"]["mri_hippocampus"].startswith("dropped")
    assert "mri_entorhinal" in bad.columns
    unk, info3 = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, SRC.verify_mri_codes(None))
    assert set(info3["feature_status"].values()) == {"unverified (codes not in dictionary)"}
    raw = d["mri_raw"].copy(); raw["ST10CV"] = raw["ST10CV"] / 1000                         # wrong unit -> implausible ICV
    implaus, info4 = SRC.mri_table(raw, syn.FS_EXTRA, SRC.verify_mri_codes(syn.datadic()))
    assert "mri_hippocampus" not in implaus.columns and "mri_entorhinal_thickness" in implaus.columns
    with pytest.raises(SRC.SchemaError):
        SRC.mri_table(d["mri_raw"].drop(columns=["EXAMDATE"]), [], {})


def test_csf_bounds_are_parsed_and_counted():
    t = pd.DataFrame({"RID": [1, 2, 3], "EXAMDATE": ["2020-01-01"] * 3, "ABETA42": [">1700", "800", "x"],
                      "PTAU": ["<8", "25.5", "30"]})
    out, info = SRC.csf_table(t)
    assert out["csf_abeta42"].tolist()[:2] == [1700.0, 800.0] and np.isnan(out["csf_abeta42"].iloc[2])
    assert info == {"values_at_assay_bound_abeta42": 1, "values_at_assay_bound_ptau": 1}


# ------------------------------------------------------------------------------------------------ estimators
def test_cox_ridge_recovers_the_direction_and_gives_monotone_risk():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (400, 2))
    t = rng.exponential(1 / (0.3 * np.exp(0.9 * x[:, 0])))
    c = rng.uniform(0.5, 4, 400)
    time, ev = np.minimum(t, c), (t <= c).astype(int)
    cox = M.CoxRidge(0.01).fit(x, time, ev)
    assert cox.coef_[0] == pytest.approx(0.9, abs=0.25) and abs(cox.coef_[1]) < 0.25
    r = cox.risk(np.array([[-1.0, 0], [0, 0], [1, 0]]), 1.0)
    assert np.all(np.diff(r) > 0) and np.all((r > 0) & (r < 1))
    assert cox.cumulative_hazard(0.0) == 0.0 and cox.cumulative_hazard(2.0) >= cox.cumulative_hazard(1.0)


def test_reference_pipelines_are_the_frozen_factory_and_fusion_records_rows():
    rng = np.random.default_rng(6)
    feats = frozen.PREDICTOR_SETS["primary"]
    X = pd.DataFrame(rng.normal(size=(150, 7)), columns=feats, index=[f"p{i}" for i in range(150)])
    X.iloc[::9, 3] = np.nan
    y = rng.binomial(1, 0.35, 150)
    a = M.sb_pipeline("lr_l2_ref", {"C": 1.0}, feats, 20261017).fit(X, y).predict_proba(X)[:, 1]
    b = frozen.make_pipeline_factory("logistic_l2", {"C": 1.0}, feats)("A3", 20261017).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)
    lf = M.LateFusionClassifier(groups={"g1": feats[:3], "g2": feats[3:]}, seed=1).fit(X.iloc[:100], y[:100])
    assert M.fitted_rows(lf) == frozenset(X.index[:100])
    assert lf.predict_proba(X.iloc[100:]).shape == (50, 2)
    ip = M.interaction_pipeline(feats, 0.1, 1).fit(X, y)
    assert ip.named_steps["poly"].n_output_features_ == 7 + 21
