"""Deliverable tables and reports of the multimodal benchmark, built ONLY from the aggregate files of one run folder
(PROTOCOL.md §10). Standard library + pandas; no model fitting; no participant data.

  python3 -m multimodal_benchmark.reporting --run-dir multimodal_benchmark/runs/<run id>

Real-data run (run_metadata.json synthetic_data == false): writes multimodal_benchmark/reports/ (experiment_registry.csv,
modality_ablation_results.csv, model_comparison.csv, calibration_results.csv, paired_contrasts.csv, heldout_results.csv,
reference_reproduction.csv) and the result-dependent reports (benchmark_report.md, final_model_selection.md,
best_model_card.md, results_determination.md, cohort_validation_status.md). A synthetic run is written only to
<run>/report_preview/ with a SYNTHETIC watermark on every file. cohort_validation_results.csv is never written while no external
cohort was evaluated: an empty table could be misread as a successful validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REF = "A|P_ALL|FS08_CORE7|lr_l2_ref|median|primary"
WATERMARK = "**SYNTHETIC SOFTWARE TEST - NOT A RESEARCH RESULT. No number below describes ADNI or any participant.**\n\n"
MATERIAL = {"roc_auc": 0.02, "brier": -0.01}


def _read(run: Path, name: str) -> pd.DataFrame:
    p = run / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _f(x, d=3):
    try:
        return "n/a" if x is None or pd.isna(x) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(r, m, d=3):
    return f"{_f(r.get(m), d)} [{_f(r.get(f'{m}_ci_low'), d)}, {_f(r.get(f'{m}_ci_high'), d)}]"


def _table(df: pd.DataFrame, cols: list) -> str:
    if df.empty:
        return "_none_\n"
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(rows) + "\n"


def material(con: pd.DataFrame, cand: str, comp: str) -> dict:
    """Prespecified materiality (PROTOCOL.md §10): an improvement is MATERIAL when its 95% interval excludes no-change in the
    favourable direction AND the point estimate passes the margin (AUROC >= +0.02 or Brier <= -0.01)."""
    c = con[(con["candidate"] == cand) & (con["comparator"] == comp)]
    out = {"available": not c.empty}
    for m, thr in MATERIAL.items():
        r = c[c["metric"] == m]
        if r.empty:
            out[m] = None; continue
        r = r.iloc[0]
        better = (r["ci_low"] > 0 and r["difference"] >= thr) if m == "roc_auc" else (r["ci_high"] < 0 and r["difference"] <= thr)
        out[m] = {"difference": r["difference"], "ci": (r["ci_low"], r["ci_high"]), "material": bool(better)}
    out["material"] = any(v and v["material"] for k, v in out.items() if k in MATERIAL)
    return out


def determination(run: Path) -> dict:
    """Mechanical application of the prespecified publication rule to one run (PROTOCOL.md §10)."""
    con, ho = _read(run, "paired_contrasts"), _read(run, "heldout_contrasts")
    sel = json.loads((run / "selection.json").read_text()) if (run / "selection.json").exists() else {}
    win = sel.get("P_ALL", {}).get("winner")
    checks = {
        "winner_vs_reference": material(con, win, REF) if win and win != REF else {"available": False, "material": False},
        "mri_increment": material(con, "A|P_MRI|FS13_MRI_CORE7|lr_l2_ref|median|primary", "A|P_MRI|FS08_CORE7|lr_l2_ref|median|primary"),
        "longitudinal_history": material(con, "C|P_LM|LM2_CURRENT_HISTORY|lr_l2_ref|median|primary",
                                         "C|P_LM|LM1_CURRENT|lr_l2_ref|median|primary"),
        "censoring_aware": material(con, "B|P_ALL|FS08_CORE7|cox_l2|median|primary", REF),
    }
    held_dir = None
    if not ho.empty and win and win != REF:
        h = ho[(ho["candidate"] == "WINNER_P_ALL") & (ho["metric"].isin(["roc_auc", "brier"]))]
        if len(h) == 2:
            d = dict(zip(h["metric"], h["difference"]))
            held_dir = bool(d["roc_auc"] > 0 or d["brier"] < 0)
    gate = _read(run, "population_gate")
    ev = gate.set_index("population")["events"].to_dict() if not gate.empty else {}
    if checks["censoring_aware"].get("material"):
        rec, why = 2, "the censoring-aware model, which uses REF-indeterminate follow-up, materially improved on the reference: this " \
                      "extends the label-determinacy question of the current manuscript"
    elif checks["winner_vs_reference"].get("material") and held_dir:
        e = ev.get("P_ALL"); big = isinstance(e, (int, float)) or str(e).isdigit()
        rec = 3 if big and int(e) >= 100 else 4
        why = ("the selected model materially improved on the frozen reference in development nested CV and in the same direction "
               "on the locked held-out set" + ("" if rec == 3 else "; fewer than 100 development events, so defer"))
    elif checks["mri_increment"].get("material") or checks["longitudinal_history"].get("material"):
        rec, why = 4, "a modality or longitudinal increment was material only within a subset; external or larger-sample " \
                      "confirmation is needed before a separate paper"
    else:
        rec, why = 1, "no model, modality, longitudinal or censoring-aware extension materially improved on the frozen reference"
    options = {1: "Keep the current label-robustness manuscript and submit it",
               2: "Revise the current manuscript to incorporate the censoring-aware result", 3: "Develop a separate multimodal "
               "prediction paper", 4: "Defer publication of the extension because the feasible evidence is insufficient"}
    return {"winner": win, "checks": checks, "heldout_same_direction": held_dir, "recommendation": rec,
            "recommendation_text": options[rec], "reason": why}


def write(run: Path) -> Path:
    meta = json.loads((run / "run_metadata.json").read_text())
    synthetic = bool(meta.get("synthetic_data", True))
    dest = run / "report_preview" if synthetic else HERE / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    mark = WATERMARK if synthetic else ""
    cr, con, reg = _read(run, "candidate_results"), _read(run, "paired_contrasts"), _read(run, "experiment_registry")
    if synthetic:
        cr = cr.assign(SYNTHETIC_TEST="not a research result")
    src = {"run_id": meta.get("run_id"), "run_metadata_sha256": hashlib.sha256((run / "run_metadata.json").read_bytes()).hexdigest()}
    cr_cols = [c for c in cr.columns if not c.endswith("_sd_repeats")] + [c for c in cr.columns if c.endswith("_sd_repeats")]
    cr.loc[cr["stage"].isin(["A", "C", "D", "E"]), cr_cols].to_csv(dest / "modality_ablation_results.csv", index=False)
    cr.loc[cr["stage"] == "B", cr_cols].to_csv(dest / "model_comparison.csv", index=False)
    cal_cols = ["candidate", "stage", "population", "feature_set", "model", "participants", "events", "brier", "log_oe",
                "calibration_intercept", "calibration_slope", "calibration_intercept_ci_low", "calibration_intercept_ci_high",
                "calibration_slope_ci_low", "calibration_slope_ci_high", "log_oe_ci_low", "log_oe_ci_high"]
    cr.reindex(columns=cal_cols).to_csv(dest / "calibration_results.csv", index=False)
    reg.to_csv(dest / "experiment_registry.csv", index=False)
    con.to_csv(dest / "paired_contrasts.csv", index=False)
    for n in ("heldout_results", "reference_reproduction", "heldout_contrasts"):
        t = _read(run, n)
        if not t.empty:
            t.to_csv(dest / f"{n}.csv", index=False)
    det = determination(run)
    sel = json.loads((run / "selection.json").read_text()) if (run / "selection.json").exists() else {}
    stab = _read(run, "selection_P_ALL")
    rr, hr = _read(run, "reference_reproduction"), _read(run, "heldout_results")
    win = det["winner"]
    wrow = cr[cr["candidate"] == win].iloc[0].to_dict() if win in set(cr.get("candidate", [])) else {}
    rrow = cr[cr["candidate"] == REF].iloc[0].to_dict() if REF in set(cr.get("candidate", [])) else {}
    head = f"{mark}Source run `{src['run_id']}` (run_metadata.json SHA-256 `{src['run_metadata_sha256']}`). Generated by " \
           f"`multimodal_benchmark/reporting.py` from aggregate files only.\n\n"
    (dest / "final_model_selection.md").write_text(
        f"# Final model selection\n\n{head}Rule (frozen, PROTOCOL.md §8): {json.dumps(sel.get('rule', {}))}.\n\n"
        f"**P_ALL winner:** `{win}`. Lowest Brier: `{sel.get('P_ALL', {}).get('best_brier')}`. Reason: {sel.get('P_ALL', {}).get('reason')}.\n\n"
        f"**P_MRI winner:** `{sel.get('P_MRI', {}).get('winner')}` ({sel.get('P_MRI', {}).get('reason', 'population not evaluated')}).\n\n"
        f"Selection hash: `{sel.get('sha256')}` (written before the held-out stage).\n\n## Decision table (P_ALL)\n\n"
        + _table(stab.round(4) if not stab.empty else stab, [c for c in ("id", "brier", "delta_brier_vs_best", "se_delta",
                 "delta_auroc_vs_best", "comparable", "calibration_adequate", "selected") if c in stab.columns]))
    (dest / "best_model_card.md").write_text(
        f"# Model card: {win}\n\n{head}| Item | Value |\n| --- | --- |\n| Intended use | research estimate of recorded any-cause "
        f"dementia within 24 months of an MCI baseline visit in ADNI-like cohorts; not for clinical use |\n| Endpoint | frozen REF "
        f"label (calendar 24 months, H +/- 90 d ascertainment, death competing; indeterminate never a non-event) |\n| Time origin "
        f"| MCI diagnosis anchor; predictors dated [-90, +14] days |\n| Development | frozen training partition, REF-labelled "
        f"(participants {wrow.get('participants')}, events {wrow.get('events')}) |\n| Nested-CV AUROC | {_ci(wrow, 'roc_auc')} |\n"
        f"| Nested-CV Brier | {_ci(wrow, 'brier')} |\n| Calibration slope / log O:E | {_ci(wrow, 'calibration_slope')} / "
        f"{_ci(wrow, 'log_oe')} |\n| Reference (CORE7 L2) AUROC / Brier | {_ci(rrow, 'roc_auc')} / {_ci(rrow, 'brier')} |\n"
        f"| Held-out (locked, once) | see heldout_results.csv |\n| External validation | none (no external cohort evaluable; "
        f"cohort_validation_status.md) |\n| Status | ESTIMATION_ONLY; not confirmatory |\n")
    (dest / "results_determination.md").write_text(
        f"# Results determination (mechanical application of PROTOCOL.md §10)\n\n{head}**Recommendation "
        f"{det['recommendation']}: {det['recommendation_text']}.** Reason: {det['reason']}.\n\n```json\n"
        f"{json.dumps(det['checks'], indent=2, default=str)}\n```\n\nHeld-out direction agrees with development: "
        f"{det['heldout_same_direction']}. This is the prespecified rule's output; the authors' interpretation is in "
        f"publication_recommendation.md.\n")
    (dest / "cohort_validation_status.md").write_text(
        f"# External cohort validation status\n\n{head}No external cohort was evaluated by this run, so no "
        f"cohort_validation_results.csv exists. See dataset_cohort_matrix.csv in the run folder and "
        f"dataset_and_modality_feasibility.md for the missing prerequisite of each cohort.\n")
    rep = [f"# Benchmark report\n\n{head}## Populations and gate\n\n", _table(_read(run, "population_gate"),
           ["population", "participants", "events", "nonevents", "status"]),
           "\n## Experiments\n\n", _table(reg.groupby(["stage", "status"]).size().reset_index(name="candidates"),
                                          ["stage", "status", "candidates"]),
           f"\n## Selected model\n\n`{win}`: AUROC {_ci(wrow, 'roc_auc')}, Brier {_ci(wrow, 'brier')}, calibration slope "
           f"{_ci(wrow, 'calibration_slope')}, log O:E {_ci(wrow, 'log_oe')}. Frozen reference: AUROC {_ci(rrow, 'roc_auc')}, "
           f"Brier {_ci(rrow, 'brier')}.\n\n## Reference reproduction (locked held-out, A3)\n\n",
           _table(rr.round(8) if not rr.empty else rr, ["model", "metric", "reproduced", "stored_A3", "abs_difference", "status"]),
           "\n## Locked held-out evaluation\n\n",
           _table(hr.round(4) if not hr.empty else hr, [c for c in ("model", "participants", "events", "roc_auc", "brier", "log_oe",
                  "calibration_slope") if c in hr.columns]),
           f"\n## Determination\n\nRecommendation {det['recommendation']}: {det['recommendation_text']} ({det['reason']}).\n",
           "\nIntervals: participant bootstrap of out-of-fold predictions, conditional on the fitted fold models (approximate; "
           "PROTOCOL.md §7). All results are estimation-only.\n"]
    (dest / "benchmark_report.md").write_text("".join(rep))
    if not synthetic:
        (dest / "STATUS.md").write_text(f"# reports/: status\n\nGenerated from real-data run `{src['run_id']}` (run_metadata.json "
                                        f"SHA-256 `{src['run_metadata_sha256']}`) by reporting.py. Do not edit by hand.\n")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    print(f"reports written to {write(a.run_dir.resolve())}")


if __name__ == "__main__":
    main()
