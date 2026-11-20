"""2 x 2 design analysis (label rule x anchoring) - pure functions, no file access. Synthetic-tested in
tests/test_design_analysis.py. The guarded real-data entry point is run_design_analysis.py (researcher only; never run by an AI
session). Protocol: protocol/ANALYSIS_PROTOCOL.md §§2-9.

Steps of `run_design` (all on in-memory tables):
  1. eligible MCI anchors of the analysis population (src/cohort/anchor_predictors.eligible_anchors);
  2. predictors at each anchor with per-predictor dates and the post-index leakage check (attach_predictors);
  3. both registered label rules at every anchor (LABEL_RULES: SIM = visit-code rule, REF = calendar three-state reference);
  4. participant-level split (60/40, stratified on the baseline REF state, fixed seed), disjointness checked;
  5. event gate on the held-out REF-determinate baseline participants (protocol §11): < 40 events -> STOP (nothing is fitted);
     40-129 -> estimation only; >= 130 -> full analysis;
  6. for each model, the full-pipeline refitting participant-cluster bootstrap (src/benchmark/design_bootstrap.py) over the four
     registered configurations, scored on the COMMON held-out baseline participants against the fixed REF target, with the
     named contrasts (CONTRASTS); the visit-weighted secondary estimand is computed by the same call and returned separately;
  7. the primary estimand (log O:E, A1 - A3, logistic model) is classified against the +/- ln 1.10 margin and the secondary
     Delta AUROC (A1 - A3) against +/- 0.02 (both approved by the PI, protocol §15 item 1).
Analyses (protocol §15; decision log 128), one run each: predictor_set primary / S-noDx / S-noMMSE-FAQ; predictor_window
primary [-90, +14] / strict [-90, 0]; endpoint any (any-cause recorded dementia) / ad (AD-type, REF only). The participant split is
stratified on the any-cause REF state in every analysis, so all of them share the same held-out participants. No result is
confirmatory (protocol §15 item 6).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd

from yoad.features.anchor_predictors import DEFAULT_WINDOWS, attach_predictors, eligible_anchors, predictor_window_summary
from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.data.visit_anchor_labels import label_calendar_reference, label_visitcode_sim2026

K = 11                                                    # counts 1..10 are suppressed in every output table
LABEL_RULES = {                                            # the visit-code rule has no etiology: the endpoint applies to REF only
    "SIM": lambda visits, anchors, cutoff, deaths, endpoint: label_visitcode_sim2026(visits, anchors),
    "REF": lambda visits, anchors, cutoff, deaths, endpoint: label_calendar_reference(visits, anchors, death_dates=deaths,
                                                                                    data_cutoff=cutoff, endpoint=endpoint),
}
ANCHOR_POLICIES = ("baseline", "all")
CONFIGS = {"A1": ("state_SIM", "baseline"), "A2": ("state_SIM", "all"), "A3": ("state_REF", "baseline"),
           "A4": ("state_REF", "all")}
REFERENCE = "A3"
CONTRASTS = {
    "label_baseline_A1_minus_A3": {"A1": 1, "A3": -1},                    # PRIMARY (log O:E, logistic model)
    "label_allvisits_A2_minus_A4": {"A2": 1, "A4": -1},
    "anchoring_REF_A4_minus_A3": {"A4": 1, "A3": -1},
    "anchoring_SIM_A2_minus_A1": {"A2": 1, "A1": -1},
    "interaction_(A2-A1)_minus_(A4-A3)": {"A2": 1, "A1": -1, "A4": -1, "A3": 1},
}
PRIMARY = ("logistic_l2", "label_baseline_A1_minus_A3", "log_oe")
PREDICTOR_SETS = {                                         # protocol §15 item 2 (decision log 128): two distinct sensitivity sets
    "primary": ["age", "female", "education_years", "mmse", "adas13", "faq", "apoe4"],
    "S-noDx": ["age", "female", "education_years", "apoe4"],                      # no diagnosis-linked scale (MMSE, ADAS-Cog13, FAQ)
    "S-noMMSE-FAQ": ["age", "female", "education_years", "adas13", "apoe4"],      # MMSE and FAQ removed, ADAS-Cog13 kept
}
PREDICTOR_WINDOWS = {                                     # protocol §15 item 4: days relative to the anchor date, inclusive
    "primary": {"post_days": 14, "windows": {k: [-90, 14] for k in DEFAULT_WINDOWS}},
    "strict": {"post_days": 0, "windows": {k: [-90, 0] for k in DEFAULT_WINDOWS}},
}
ENDPOINTS = ("any", "ad")                                 # protocol §15 item 3: any-cause primary, AD-type sensitivity
DEFAULT_CONFIG = {
    "seed": 20261017, "train_fraction": 0.6, "B": 1000, "alpha": 0.05, "u_days": 90,
    "predictor_set": "primary", "predictor_window": "primary", "endpoint": "any",
    "models": {"logistic_l2": {"C": 1.0},
               "xgboost": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05}},
    "metrics": ["roc_auc", "average_precision", "brier", "log_oe", "calibration_intercept", "calibration_slope"],
    "min_eval_events": 40, "full_analysis_events": 130, "materiality_margin_log": float(np.log(1.10)),
    "materiality_margin_auroc": 0.02,
}


def sup(n):
    n = int(n)
    return n if n == 0 or n >= K else f"<{K}"


def label_anchors(visits: pd.DataFrame, anchors: pd.DataFrame, data_cutoff, death_dates: dict | None = None,
                  endpoint: str = "any") -> pd.DataFrame:
    """Adds state_<rule> and reason_<rule> for every registered rule, on identical anchors, plus split_stratum (the any-cause
    REF state, whatever the endpoint) so that every analysis splits participants identically."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}")
    a = anchors[["participant_id", "anchor_month", "anchor_date"]]
    out = anchors.copy()
    for rule, fn in LABEL_RULES.items():
        lab = fn(visits, a, data_cutoff, death_dates, endpoint)
        if len(lab) != len(a) or not (lab["anchor_month"].to_numpy() == a["anchor_month"].to_numpy()).all():
            raise AssertionError(f"rule {rule} did not return one label per anchor in order")
        out[f"state_{rule}"] = lab["state"].to_numpy(); out[f"reason_{rule}"] = lab["reason"].to_numpy()
    out["split_stratum"] = (out["state_REF"] if endpoint == "any" else
                            LABEL_RULES["REF"](visits, a, data_cutoff, death_dates, "any")["state"].to_numpy())
    return out


def split_participants(rows: pd.DataFrame, seed: int, train_fraction: float) -> tuple[set, set]:
    """Participant-level split stratified on the baseline any-cause REF state (split_stratum; state_REF if absent)."""
    from sklearn.model_selection import train_test_split
    col = "split_stratum" if "split_stratum" in rows.columns else "state_REF"
    base = rows[rows["is_baseline"]][["participant_id", col]].drop_duplicates("participant_id")
    tr, te = train_test_split(base["participant_id"].to_numpy(), train_size=train_fraction, random_state=seed,
                              stratify=base[col].to_numpy())
    tr, te = set(tr), set(te)
    if tr & te:
        raise AssertionError("participant in both partitions")
    return tr, te


def classify_interval(lo, hi, margin):
    """Protocol §8 rule; margins approved by the PI (§15 item 1): +/- ln 1.10 for log O:E, +/- 0.02 for AUROC."""
    if lo is None or hi is None:
        return "not estimable"
    if lo > margin or hi < -margin:
        return "material effect (95% CI wholly outside the margin)"
    if lo >= -margin and hi <= margin:
        return "no material effect (95% CI wholly inside the margin)"
    return "inconclusive (95% CI crosses a margin boundary)"


def make_pipeline_factory(model_id: str, params: dict, features: list):
    from yoad.benchmark.config import load_config
    from yoad.benchmark.data import FeatureInfo
    from yoad.benchmark.pipeline import build_pipeline
    feats = [FeatureInfo(f, "clinical", "numeric") for f in features]
    cfg = load_config({"experiment_id": "design_2x2", "model": {"id": model_id, "params": dict(params)},
                       "preprocessing": {"numeric_imputation": "median", "scaling": "standard"}, "metrics": ["roc_auc"]})
    return lambda name, seed: build_pipeline(cfg, feats, dict(params), seed)


def _counts(rows: pd.DataFrame, train: set, test: set) -> pd.DataFrame:
    recs = []
    for part, ids in (("train", train), ("held_out", test)):
        r = rows[rows["participant_id"].isin(ids)]
        for rule in LABEL_RULES:
            for anch in ANCHOR_POLICIES:
                t = r if anch == "all" else r[r["is_baseline"]]
                s = t[f"state_{rule}"]
                recs.append({"partition": part, "rule": rule, "anchoring": anch, "n_anchors": sup(len(t)),
                             "n_participants": sup(t["participant_id"].nunique()), "n_event": sup((s == EVENT).sum()),
                             "n_nonevent": sup((s == NONEVENT).sum()), "n_not_labelled": sup((~s.isin([EVENT, NONEVENT])).sum())})
    return pd.DataFrame(recs)


def _reasons(rows: pd.DataFrame) -> pd.DataFrame:
    t = []
    for rule in LABEL_RULES:
        for anch in ANCHOR_POLICIES:
            r = rows if anch == "all" else rows[rows["is_baseline"]]
            g = r.groupby([f"state_{rule}", f"reason_{rule}"]).size().reset_index(name="n_anchors")
            g.columns = ["state", "reason", "n_anchors"]; g.insert(0, "anchoring", anch); g.insert(0, "rule", rule)
            t.append(g)
    out = pd.concat(t, ignore_index=True); out["n_anchors"] = out["n_anchors"].map(sup)
    return out


def run_design(visits: pd.DataFrame, measurements: pd.DataFrame, static: pd.DataFrame, data_cutoff,
               death_dates: dict | None = None, config: dict | None = None, synthetic: bool = False) -> dict:
    """Returns {"tables": {name: DataFrame (aggregate only)}, "metadata": dict}."""
    from yoad.benchmark.design_bootstrap import design_contrast_bootstrap
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["predictor_window"] not in PREDICTOR_WINDOWS or cfg["endpoint"] not in ENDPOINTS:
        raise ValueError("predictor_window must be one of %s and endpoint one of %s" % (list(PREDICTOR_WINDOWS), ENDPOINTS))
    cfg.update(PREDICTOR_WINDOWS[cfg["predictor_window"]])      # windows and post_days come only from the named window
    t0 = time.time()
    meta = {"synthetic_data": bool(synthetic), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "data_cutoff": str(pd.Timestamp(data_cutoff).date()), "python": platform.python_version(),
            "configs": CONFIGS, "reference": REFERENCE, "contrasts": CONTRASTS, "primary": list(PRIMARY),
            "materiality_margin_status": "approved by the PI (ANALYSIS_PROTOCOL.md §15 item 1; decision log 127-128)",
            "analysis": {"predictor_set": cfg["predictor_set"], "predictor_window": cfg["predictor_window"],
                         "endpoint": cfg["endpoint"],
                         "role": "primary" if (cfg["predictor_set"], cfg["predictor_window"], cfg["endpoint"])
                                 == ("primary", "primary", "any") else "sensitivity"},
            "confirmatory": False, "confirmation_set": "deferred (protocol §15 item 6): results are not confirmatory"}
    import sklearn
    meta.update(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)
    windows = {k: tuple(v) for k, v in cfg["windows"].items()}
    features = PREDICTOR_SETS[cfg["predictor_set"]] if isinstance(cfg["predictor_set"], str) else list(cfg["predictor_set"])

    anchors, flow = eligible_anchors(visits, data_cutoff, u_days=cfg["u_days"], death_dates=death_dates)
    rows, mflow = attach_predictors(anchors, measurements, static, windows=windows, post_days=cfg["post_days"],
                                    data_cutoff=data_cutoff)
    rows = label_anchors(visits, rows, data_cutoff, death_dates, cfg["endpoint"])
    train, test = split_participants(rows, cfg["seed"], cfg["train_fraction"])
    tables = {"flow": pd.DataFrame([{"step": k, "n": sup(v)} for k, v in {**flow, **mflow}.items()]),
              "label_counts": _counts(rows, train, test), "label_reasons": _reasons(rows),
              "predictor_windows": predictor_window_summary(rows, windows).assign(
                  n_anchors=lambda d: d.n_anchors.map(sup), n_available=lambda d: d.n_available.map(sup))}

    train_anchors = rows[rows["participant_id"].isin(train)].copy()
    held = rows[rows["participant_id"].isin(test)]
    eval_base = held[held["is_baseline"] & held["state_REF"].isin([EVENT, NONEVENT])].copy()
    eval_base["y_ref"] = (eval_base["state_REF"] == EVENT).astype(int)
    eval_vis = held[held["state_REF"].isin([EVENT, NONEVENT])].copy()
    eval_vis["y_ref"] = (eval_vis["state_REF"] == EVENT).astype(int)
    n_ev = int(eval_base["y_ref"].sum())
    gate = ("STOP" if n_ev < cfg["min_eval_events"] else
            "ESTIMATION_ONLY" if n_ev < cfg["full_analysis_events"] else "FULL_ANALYSIS")
    meta.update(event_gate=gate, eval_baseline_events=sup(n_ev), eval_baseline_participants=sup(len(eval_base)),
                eval_visit_anchors=sup(len(eval_vis)), eval_visit_participants=sup(eval_vis["participant_id"].nunique()),
                train_participants=sup(len(train)))
    results = {}
    if gate == "STOP":
        meta["stopped"] = f"fewer than {cfg['min_eval_events']} REF events among held-out determinate baseline participants"
    else:
        for model_id, params in cfg["models"].items():
            try:
                r = design_contrast_bootstrap(train_anchors, eval_base, "y_ref", CONFIGS, features,
                                              make_pipeline_factory(model_id, params, features), REFERENCE,
                                              metrics=cfg["metrics"], B=cfg["B"], seed=cfg["seed"], alpha=cfg["alpha"],
                                              contrasts=CONTRASTS, visit_eval_frame=eval_vis, visit_target="y_ref")
                results[model_id] = r
            except Exception as e:                                        # recorded; other models still run
                results[model_id] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        est, dif, vest, vdif, fails = [], [], [], [], []
        for model_id, r in results.items():
            if "error" in r:
                fails.append({"model": model_id, "scope": "whole run", "replicate": None, "error": r["error"]}); continue
            est += [{"model": model_id, **e} for e in r["estimates"]]
            dif += [{"model": model_id, **{k: v for k, v in d.items() if k != "coefficients"}} for d in r["paired_differences"]]
            vest += [{"model": model_id, **e} for e in r["visit_weighted"]["estimates"]]
            vdif += [{"model": model_id, **{k: v for k, v in d.items() if k != "coefficients"}}
                     for d in r["visit_weighted"]["paired_differences"]]
            fails += [{"model": model_id, "scope": "baseline", **f} for f in r["failures"]]
            fails += [{"model": model_id, "scope": "visit_weighted", **f} for f in r["visit_weighted"]["failures"]]
            meta[f"{model_id}_B_used"] = r["B_used"]; meta[f"{model_id}_visit_weighted_B_used"] = r["visit_weighted"]["B_used"]
        tables.update(baseline_estimates=pd.DataFrame(est), baseline_contrasts=pd.DataFrame(dif),
                      visit_weighted_estimates=pd.DataFrame(vest), visit_weighted_contrasts=pd.DataFrame(vdif),
                      failures=pd.DataFrame(fails, columns=["model", "scope", "replicate", "error"]))
        pm, pc, pmet = PRIMARY
        prim = [d for d in dif if d["model"] == pm and d["contrast"] == pc and d["metric"] == pmet]
        if prim:
            p = prim[0]
            meta["primary"] = {"estimand": "log O:E, A1 - A3, logistic model, held-out REF-determinate baseline participants",
                               "difference": p["difference"], "ci_low": p["ci_low"], "ci_high": p["ci_high"],
                               "margin": cfg["materiality_margin_log"],
                               "classification": classify_interval(p["ci_low"], p["ci_high"], cfg["materiality_margin_log"]),
                               "event_gate": gate}
        sec = [d for d in dif if d["model"] == pm and d["contrast"] == pc and d["metric"] == "roc_auc"]
        if sec:
            s = sec[0]
            meta["secondary_auroc"] = {"estimand": "AUROC, A1 - A3, logistic model, same held-out participants",
                                       "difference": s["difference"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                                       "margin": cfg["materiality_margin_auroc"],
                                       "classification": classify_interval(s["ci_low"], s["ci_high"],
                                                                           cfg["materiality_margin_auroc"])}
    meta["seconds"] = round(time.time() - t0, 1)
    return {"tables": tables, "metadata": meta}


# ------------------------------------------------------------------ ADNI table mappers (pure; used only by the guarded runner)
# Column names follow experiments/master/e02_anchor_features.py; verify them against the local release before the run [U].
ADNI_MEASURES = {"mmse": ("MMSCORE", "VISDATE"), "adas13": ("TOTAL13", "VISDATE"), "faq": ("FAQTOTAL", "VISDATE")}


def adni_measurements(tables: dict) -> pd.DataFrame:
    """tables: {"mmse": df, "adas13": df, "faq": df} with RID, the date column and the score. Negative codes -> missing."""
    out = []
    for var, df in tables.items():
        col, dcol = ADNI_MEASURES[var]
        v = pd.to_numeric(df[col], errors="coerce")
        out.append(pd.DataFrame({"participant_id": pd.to_numeric(df["RID"], errors="coerce"), "variable": var,
                                 "date": df[dcol], "value": v.where(v >= 0)}))
    return pd.concat(out, ignore_index=True).dropna(subset=["participant_id"])


def adni_static(adsl: pd.DataFrame, apoe: pd.DataFrame, death_imputed: str = "as_recorded") -> tuple[pd.DataFrame, dict]:
    """Static predictors from ADSL (SUBJID, AGE, SEX, EDUC, DTHDT) and APOERES (RID, GENOTYPE). Returns (static, death_dates)."""
    a = adsl.rename(columns={"SUBJID": "RID"}).copy()
    a["RID"] = pd.to_numeric(a["RID"], errors="coerce")
    a = a.dropna(subset=["RID"]).drop_duplicates("RID")
    g = apoe.copy(); g["RID"] = pd.to_numeric(g["RID"], errors="coerce")
    g = g.dropna(subset=["RID", "GENOTYPE"]).drop_duplicates("RID").set_index("RID")["GENOTYPE"]
    st = pd.DataFrame({"participant_id": a["RID"].to_numpy(), "baseline_age": pd.to_numeric(a["AGE"], errors="coerce").to_numpy(),
                       "female": (a["SEX"] == "Female").astype(float).where(a["SEX"].notna()).to_numpy(),
                       "education_years": pd.to_numeric(a["EDUC"], errors="coerce").to_numpy()})
    st["apoe4"] = st["participant_id"].map(g).astype(str).str.count("4").clip(upper=2).astype(float).where(
        st["participant_id"].map(g).notna())
    from yoad.experiments.visit_anchor_audit.audit_lib import adsl_death_dates
    deaths, _ = adsl_death_dates(adsl, death_imputed)
    return st, deaths


def adni_dem_type(dxsum: pd.DataFrame) -> pd.DataFrame:
    """Dementia etiology per participant-visit for the AD-type endpoint sensitivity (protocol §15 item 3), with the rule already
    used in experiments/master: ADNI1 DXAD = 1 -> AD, DXOTHDEM = 1 -> OTHER; later phases DXDDUE 1 -> AD, 2 -> OTHER; anything
    else UNKNOWN (never AD). Several rows of one visit that disagree -> UNKNOWN. Returns participant_id, visit_month, dem_type."""
    from yoad.experiments.visit_anchor_audit.audit_lib import viscode_month
    need = ["RID", "VISCODE2", "PHASE", "DXAD", "DXOTHDEM", "DXDDUE"]
    miss = [c for c in need if c not in dxsum.columns]
    if miss:
        raise ValueError(f"DXSUM lacks {miss}, needed for the AD-type endpoint; verify the release")
    num = lambda c: pd.to_numeric(dxsum[c], errors="coerce")
    adni1 = dxsum["PHASE"].astype(str).str.upper().eq("ADNI1")
    typ = np.where(adni1, np.select([num("DXAD") == 1, num("DXOTHDEM") == 1], ["AD", "OTHER"], "UNKNOWN"),
                   np.select([num("DXDDUE") == 1, num("DXDDUE") == 2], ["AD", "OTHER"], "UNKNOWN"))
    d = pd.DataFrame({"participant_id": pd.to_numeric(dxsum["RID"], errors="coerce"),
                      "visit_month": dxsum["VISCODE2"].map(viscode_month), "dem_type": typ}).dropna(subset=["participant_id", "visit_month"])
    d["visit_month"] = d["visit_month"].astype(int)
    g = d.groupby(["participant_id", "visit_month"])["dem_type"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "UNKNOWN")
    return g.reset_index()
