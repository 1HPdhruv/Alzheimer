import numpy as np
import pandas as pd
import pytest
from tests.synthetic_baseline import raw_tables, MRI
from yoad.data import simkhada_dataset as sd
from yoad.experiments.simkhada_baseline import baseline_lib as bl

def test_build_dataset_reproduces_reference_construction():
    s, flow, info = sd.build_dataset(raw_tables())
    got = {(int(p), int(m)): int(y) for p, m, y in zip(s.participant_id, s.anchor_month, s.y)}
    # P1 bl: event via the undated m12 dementia (kept for labels); P1 m06 dropped by the -4 MMSE code (complete case);
    # P2: stable at bl (MCI at m24); later P2 anchors lack m+24 visits; P3 excluded (CN at m06); P4 advisory-excluded
    assert got == {(1, 0): 1, (2, 0): 0}
    assert flow["advisory_participants_excluded"] == 1 and flow["dxsum_rows_undated_kept_for_labels"] == 1
    assert flow["mmse_MMSCORE_negative_codes_set_missing"] == 1 and flow["mmse_duplicate_visit_rows_dropped"] == 1
    assert flow["label_EXCLUDED_reverted_to_cn"] >= 1
    r1 = s[s.participant_id == 1].iloc[0]
    assert abs(r1.AGE - (pd.Timestamp("2010-01-10") - pd.Timestamp("1940-01-10")).days / 365.25) < 1e-9
    assert r1.PTGENDER == 1 and r1.PTHAND == 0 and r1.APOE4 == 1 and r1.MMSCORE == 28
    assert info["features"]["MRI"] == [c for c in MRI] and "OTHERCOL" not in info["features"]["MRI"]
    assert info["mri_counts"] == {"SV": 2, "CV": 1, "SA": 1, "TA": 1, "TS": 1} and info["mri_counts_match_paper"] is False
    num, cat = sd.feature_columns(info, "ALL")
    assert cat == ["PTGENDER", "PTHAND"] and len(num) == 6 + 3 + len(MRI)
    assert sd.feature_columns(info, "COG") == (sd.COG, [])

def test_build_dataset_rejects_missing_columns_and_honours_explicit_mri_list():
    t = raw_tables(); t["cdr"] = t["cdr"].drop(columns=["CDRSB"])
    with pytest.raises(ValueError, match="CDRSB"):
        sd.build_dataset(t)
    s, _, info = sd.build_dataset(raw_tables(), mri_explicit=["ST1SV", "ST10CV"])
    assert info["features"]["MRI"] == ["ST1SV", "ST10CV"] and info["mri_selection"] == "explicit list"
    with pytest.raises(ValueError):
        sd.build_dataset(raw_tables(), mri_explicit=["ST999SV"])
    s2, flow2, _ = sd.build_dataset(raw_tables(), exclude_advisory=False)
    assert flow2["advisory_participants_excluded"] == 0 and 4 in set(s2.participant_id)

def test_baseline_cohort_with_complete_features_is_kept_and_traced():
    s, flow, _ = sd.build_dataset(raw_tables())
    b = sd.restrict_to_baseline(s, flow)
    assert sorted(b.participant_id) == [1, 2] and (b.anchor_month == 0).all()
    assert flow["m0_mci_anchors_before_labelling"] == 3 and flow["m0_anchors_labelled"] == 3       # P4 advisory-excluded
    assert (flow["m0_labels_n_event"], flow["m0_labels_n_nonevent"], flow["m0_labels_n_excluded_visitcode_rule"]) == (1, 1, 1)
    assert flow["m0_label_EXCLUDED_reverted_to_cn"] == 1
    assert flow["m0_determinate_before_complete_case"] == flow["m0_determinate_complete_case"] == 2
    assert flow["m0_events_complete_case"] == 1 and flow["m0_missing_MMSCORE"] == 0 and flow["m0_missing_all_mri"] == 0
    assert flow["samples_baseline_only"] == 2

def test_baseline_cohort_lost_to_missing_same_visit_predictors_fails_fast():
    t = raw_tables()
    t["mmse"] = t["mmse"].replace({"VISCODE2": {"bl": "sc"}})          # MMSE recorded at screening, not at the bl visit
    t["ucsffsx7"] = t["ucsffsx7"].replace({"VISCODE2": {"bl": "scmri"}})
    s, flow, _ = sd.build_dataset(t)
    assert flow["m0_determinate_before_complete_case"] == 2 and flow["m0_determinate_complete_case"] == 0
    assert flow["m0_missing_MMSCORE"] == 2 and flow["m0_missing_CDRSB"] == 0
    assert flow["m0_missing_any_mri"] == flow["m0_missing_all_mri"] == 2
    with pytest.raises(sd.EmptyCohortError, match="m0_determinate_complete_case"):
        sd.restrict_to_baseline(s, flow)
    from yoad.experiments.visit_anchor_audit.audit_lib import flow_table
    ft = flow_table(flow)                                            # still exportable; counts 1..10 suppressed
    assert set(ft.loc[ft.step.str.startswith("m0_"), "n"].dropna()) <= {0, "<11"}

def test_baseline_anchor_with_indeterminate_outcome_is_reported_not_labelled():
    t = raw_tables()
    t["dxsum"] = t["dxsum"][(t["dxsum"].RID == 2) & t["dxsum"].VISCODE2.isin(["bl", "m06"])]    # no follow-up near 24 months
    s, flow, _ = sd.build_dataset(t, label_rule="REF", data_cutoff="2020-01-01")
    assert flow["m0_mci_anchors_before_labelling"] == flow["m0_anchors_labelled"] == 1
    assert flow["m0_labels_n_indeterminate_other"] == 1 and flow["m0_labels_n_event"] == flow["m0_labels_n_nonevent"] == 0
    assert flow["m0_determinate_before_complete_case"] == 0
    with pytest.raises(sd.EmptyCohortError, match="m0_determinate_before_complete_case"):
        sd.restrict_to_baseline(s, flow)

def samples(n_part=90, seed=0):
    """Visit-level synthetic samples: several visits per participant, mixed labels within some participants."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_part):
        z = rng.normal(); sex = int(rng.random() < 0.5); hand = int(rng.random() < 0.1)
        for v in range(int(rng.integers(1, 4))):
            p = 1 / (1 + np.exp(-(1.4 * z + 0.3 * v - 0.6)))
            rows.append({"participant_id": i, "anchor_month": 6 * v, "y": int(rng.random() < p),
                         "f1": z + rng.normal(0, .5), "f2": rng.normal(), "f3": z * 0.5 + rng.normal(0, .8),
                         "f4": rng.normal() if rng.random() > 0.05 else np.nan, "PTGENDER": sex, "PTHAND": hand})
    return pd.DataFrame(rows)

def test_pipeline_imputes_inside_training_and_scales_only_for_linear_models():
    d = samples()
    X = d[NUM + CAT]
    for a in bl.ALGORITHMS:
        pipe = bl.make_pipeline(a, "none", NUM, CAT, 11).fit(X.iloc[:150], d.y.iloc[:150])
        assert pipe.named_steps["fit_guard"].fitted_rows_ == frozenset(X.index[:150])
        imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
        assert imp.return_scaled == (a in bl.LINEAR)
        out = pipe.named_steps["prep"].transform(X.iloc[150:])
        assert not np.isnan(out).any() and out.shape[1] == len(NUM) + len(CAT)       # categoricals last
        s = pipe.decision_function(X.iloc[150:]) if a == "svc" else pipe.predict_proba(X.iloc[150:])[:, 1]
        assert len(s) == len(X) - 150

def test_grouped_test_split_keeps_participants_apart():
    d = samples()
    tr, te = bl.grouped_test_split(d.y.to_numpy(), d.participant_id.to_numpy(), 11)
    assert not set(d.participant_id.iloc[tr]) & set(d.participant_id.iloc[te])
    assert 0.12 < len(set(d.participant_id.iloc[te])) / d.participant_id.nunique() < 0.28
    sp = bl.repeated_sgkf(d.y.to_numpy(), d.participant_id.to_numpy(), 3, 2, 11)
    assert len(sp) == 6 and all(not set(d.participant_id.iloc[a]) & set(d.participant_id.iloc[b]) for a, b in sp)

def test_final_model_never_uses_test_rows_for_search_threshold_or_calibration(monkeypatch):
    d = samples()
    seen = {}
    real = bl.select_threshold
    monkeypatch.setattr(bl, "select_threshold", lambda m, v, y, p: (seen.setdefault("n", len(y)), real(m, v, y, p))[1])
    r = bl.final_model(d, NUM, CAT, "logistic", "none", **SMALL)
    assert seen["n"] == r["n_train_samples"]                                     # threshold chosen on training OOF only
    # corrupting the test rows changes nothing that is fitted: same search result and threshold
    tr, te = bl.grouped_test_split(d.y.to_numpy(), d.participant_id.to_numpy(), 11)
    bad = d.copy(); bad.loc[bad.index[te], NUM] = 1e6       # features only: labels drive the stratified split itself
    r2 = bl.final_model(bad, NUM, CAT, "logistic", "none", **SMALL)
    assert r2["best_params"] == r["best_params"] and r2["threshold"] == r["threshold"] and r2["search_cv_best"] == r["search_cv_best"]
    assert r2["test_metrics"] != r["test_metrics"]
    b = r["test_bootstrap"]
    assert set(b.unit) == {"sample", "participant"} and (b.B_used + b.B_skipped_single_class == 60).all()
    assert {"balanced_accuracy", "roc_auc", "pr_auc", "brier", "calibration_slope"} <= set(r["test_metrics"])
    r3 = bl.final_model(d, NUM, CAT, "logistic", "none", **SMALL)                 # deterministic
    assert r3["test_metrics"] == r["test_metrics"] and r3["threshold"] == r["threshold"]

def test_bootstrap_units_differ_and_point_metrics_are_sane():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 4); y = np.repeat(rng.integers(0, 2, 40), 4); p = np.clip(0.35 * y + 0.65 * np.repeat(rng.random(40), 4) * 0.8 + 0.1 * rng.random(160), 0, 1)
    pm = bl.point_metrics(y, p, 0.5)
    assert pm["balanced_accuracy"] == pytest.approx((pm["sensitivity"] + pm["specificity"]) / 2)
    s = bl.bootstrap(y, p, 0.5, g, "sample", 300, 3).set_index("metric")
    c = bl.bootstrap(y, p, 0.5, g, "participant", 300, 3).set_index("metric")
    w = lambda t: t.loc["roc_auc", "ci_high"] - t.loc["roc_auc", "ci_low"]
    assert w(c) > w(s)                                       # clustered samples: participant intervals are wider

def test_smotenc_runs_only_while_fitting_and_keeps_categoricals_binary():
    pytest.importorskip("imblearn")
    d = samples(120)
    X, y = d[NUM + CAT], d.y.to_numpy()
    pipe = bl.make_pipeline("xgboost", "smotenc", NUM, CAT, 11).fit(X, y)
    m = pipe.named_steps["model"]
    assert m.n_resampled_rows_ > m.n_input_rows_ and m.resampled_class_counts_[0] == m.resampled_class_counts_[1]
    Xr, yr = m.sampler.fit_resample(pipe.named_steps["prep"].transform(X), y)
    assert set(np.unique(Xr[:, -2:])) <= {0.0, 1.0}                              # nominal columns stay valid categories
    assert len(pipe.predict_proba(X)) == len(X)                                  # prediction never resamples
    t = bl.make_pipeline("logistic", "smotenc_tomek", NUM, CAT, 11).fit(X, y).named_steps["model"]
    assert t.n_resampled_rows_ <= 2 * max(np.bincount(y))

NUM, CAT = ["f1", "f2", "f3", "f4"], ["PTGENDER", "PTHAND"]
SMALL = dict(n_iter=3, outer=(3, 1), B=60)
