"""Synthetic end-to-end tests of the benchmark engine, the locked held-out stage and the report builder."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.multimodal import cohort as C, engine as E, feasibility as F, registry as R, reporting as REP, sources as SRC
from tests.multimodal import synthetic_cohort as syn


def _small_plan(monkeypatch):
    monkeypatch.setattr(R, "OUTER_REPEATS", 2)
    monkeypatch.setattr(R, "MIN_EVENTS", 10); monkeypatch.setattr(R, "MIN_NONEVENTS", 10)
    monkeypatch.setattr(R, "STAGE_A_SETS", {"P_ALL": ("FS02_COG", "FS08_CORE7", "FS08b_TRANSPORT", "FS16_CORE7_COGX"),
                                             "P_MRI": ("FS08_CORE7", "FS13_MRI_CORE7")})
    monkeypatch.setattr(R, "STAGE_B_SETS", (("P_ALL", "FS08_CORE7"), ("P_MRI", "FS13_MRI_CORE7")))
    monkeypatch.setattr(R, "STAGE_B_MODELS", ("lr_enet", "cox_l2", "late_fusion", "svm_rbf", "xgb_tuned", "xgb_tuned_platt"))
    monkeypatch.setattr(R, "SIMKHADA_SETS", (("P_ALL", "FS18n_SIMKHADA_NOMRI"),))
    monkeypatch.setattr(R, "IMPUTATION_VARIANTS", ("complete_case", "iterative"))
    monkeypatch.setattr(R, "LANDMARK_MODELS", ("lr_l2_ref",))
    models = {k: dict(v) for k, v in R.MODELS.items()}
    for v in models.values():
        if "n_iter" in v:
            v["n_iter"] = 2
    models["xgb_simkhada"]["inner_splits"] = 3
    monkeypatch.setattr(R, "MODELS", models)


@pytest.fixture(scope="module")
def cohort_data():
    d = syn.make(420, seed=7)
    mri, _ = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, SRC.verify_mri_codes(syn.datadic()))
    wide = {"mri": (mri, [c for c in mri.columns if c not in ("participant_id", "date")])}
    fr = C.build_frames(d["visits"], d["measurements"], d["static"], d["cutoff"], d["death_dates"], d["extra_long"], wide)
    rows = fr["rows"]["primary"]
    dev = rows[rows["is_baseline"] & (rows["partition"] == "train")]
    return {"d": d, "fr": fr, "lm": C.landmark_frame(rows, d["measurements"], d["cutoff"])[0],
            "sv": C.survival_data(d["visits"], dev, d["cutoff"], d["death_dates"]), "dev": dev}


def test_end_to_end_reproduces_frozen_A3_and_stays_aggregate(monkeypatch, cohort_data, tmp_path):
    _small_plan(monkeypatch)
    c = cohort_data
    folds = C.outer_folds(c["dev"])
    bench = E.run_benchmark(c["fr"]["rows"], folds, c["lm"], c["sv"], syn.FS_EXTRA, n_jobs=1, B=40, log=lambda s: None)
    t = bench["tables"]
    reg = t["experiment_registry"]
    assert (reg.loc[reg["status"].isin(["completed", "failed"]), "status"] == "completed").all() and t["failures"].empty
    assert {"A", "B", "C", "D", "E"} <= set(t["candidate_results"]["stage"])
    assert bench["selection"]["P_ALL"]["winner"] in set(t["candidate_results"]["candidate"])
    fams = set(t["paired_contrasts"]["family"])
    assert {"modality_vs_core7", "mri_increment", "model_vs_reference_lr", "longitudinal", "imputation",
            "timing_sensitivity", "calibration_effect", "simkhada_vs_reference"} <= fams
    cc = t["candidate_results"].set_index("candidate")
    assert cc.loc["D|P_ALL|FS08_CORE7|lr_l2_ref|complete_case|primary", "complete"]
    # frozen 2 x 2 on the same synthetic inputs gives the stored A3 point estimates
    d = c["d"]
    res = frozen.run_design(d["visits"], d["measurements"], d["static"], d["cutoff"], death_dates=d["death_dates"],
                            config={"B": 2, "min_eval_events": 1, "full_analysis_events": 10 ** 6}, synthetic=True)
    est = res["tables"]["baseline_estimates"]
    stored = {m: {k: float(est[(est.model == key) & (est.config == "A3") & (est.metric == k)]["point"].iloc[0])
                  for k in E.HELDOUT_METRICS} for m, key in (("lr_l2_ref", "logistic_l2"), ("xgb_ref", "xgboost"))}
    ho = E.run_heldout(c["fr"]["rows"], bench, stored, B=40)
    rr = ho["tables"]["reference_reproduction"]
    assert (rr["status"] == "reproduced").all(), rr.to_string()
    assert set(ho["tables"]["heldout_models"]["status"]) == {"evaluated"}
    bc = F.baseline_characteristics(c["fr"]["rows"]["primary"])
    assert set(bc["subset"]) >= {"eligible", "REF_labelled", "REF_event"} and bc["participants"].astype(str).ne("").all()
    tables = {**t, **ho["tables"], "baseline_characteristics": bc,
              "modality_availability": F.modality_availability(c["fr"]["rows"]["primary"]),
              "availability_by_phase": F.availability_by_phase(c["fr"]["rows"]["primary"])}
    F.assert_aggregate_only(tables)
    with pytest.raises(ValueError):
        F.assert_aggregate_only({"bad": pd.DataFrame({"participant_id": [1]})})
    with pytest.raises(ValueError):
        F.assert_aggregate_only({"bad": pd.DataFrame({"events": [3]})})
    tp = E.transport_parameters(bench)
    assert tp["features"] == R.resolve("FS08b_TRANSPORT") and len(tp["coefficients"]) == 6
    # reports from a synthetic run folder: preview only, watermarked
    run = tmp_path / "run"; run.mkdir()
    for n, df in tables.items():
        df.to_csv(run / f"{n}.csv", index=False)
    (run / "selection.json").write_text(json.dumps(bench["selection"]))
    (run / "run_metadata.json").write_text(json.dumps({"run_id": "synthetic-test", "synthetic_data": True}))
    dest = REP.write(run)
    assert dest == run / "report_preview"
    assert REP.WATERMARK.strip() in (dest / "benchmark_report.md").read_text()
    assert not (dest / "cohort_validation_results.csv").exists()
    det = REP.determination(run)
    assert det["recommendation"] in (1, 2, 3, 4)


def test_parallel_and_serial_runs_are_identical(monkeypatch, cohort_data):
    _small_plan(monkeypatch)
    monkeypatch.setattr(R, "STAGE_A_SETS", {"P_ALL": ("FS08_CORE7",)})
    monkeypatch.setattr(R, "STAGE_B_SETS", (("P_ALL", "FS08_CORE7"),))
    monkeypatch.setattr(R, "STAGE_B_MODELS", ("lr_enet", "xgb_tuned"))
    monkeypatch.setattr(R, "SIMKHADA_SETS", ()); monkeypatch.setattr(R, "IMPUTATION_VARIANTS", ())
    monkeypatch.setattr(R, "LANDMARK_SETS", {})
    c = cohort_data
    folds = C.outer_folds(c["dev"])
    a = E.run_benchmark(c["fr"]["rows"], folds, None, c["sv"], syn.FS_EXTRA, n_jobs=1, B=20, log=lambda s: None)
    b = E.run_benchmark(c["fr"]["rows"], folds, None, c["sv"], syn.FS_EXTRA, n_jobs=2, B=20, log=lambda s: None)
    for cid in a["cands"]:
        assert np.array_equal(a["cands"][cid]["P"], b["cands"][cid]["P"], equal_nan=True)
    pd.testing.assert_frame_equal(a["tables"]["candidate_results"], b["tables"]["candidate_results"])
