"""Frozen registry of the multimodal benchmark v1 (multimodal_benchmark/PROTOCOL.md §§3-8).

Feature groups and sets, analysis populations, candidate models with bounded search grids, the structured experiment matrix, the
prespecified paired contrasts, validation settings and the winner-selection constants. Pure data and pure functions; no file
access. Everything that defines the scientific plan enters `config_sha256()`; the guarded runner refuses to run when that hash
differs from FREEZE.json. Do not edit without a protocol decision (a change is a new protocol version, never an in-place edit).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math

from yoad.experiments.visit_anchor_audit import design_lib as frozen

# ------------------------------------------------------------------------------------------------ validation settings
SEED = 20261017                    # the frozen 2 x 2 seed; outer repeat r uses SEED + 1000 * (r + 1)
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS = 5, 3, 3
BOOT_B, ALPHA = 2000, 0.05
K_SUPPRESS = 11                    # counts 1..10 suppressed in every output table
MIN_EVENTS, MIN_NONEVENTS = 50, 50  # a population/experiment is fitted only above both (>= 10 events per outer fold)
FULL_FROM_EVENTS = 130             # below this the benchmark is ESTIMATION_ONLY (mirrors the frozen 130-event full-analysis gate)
MAX_MISSING_FRACTION = 0.30        # a NEW feature missing in > 30% of an experiment's labelled participants is dropped
INTERACTION_MAX_BASE = 15          # pairwise-interaction model only for <= 15 base features
LANDMARK_MIN_MONTH = 12            # landmark = earliest eligible anchor at protocol month >= 12
WINDOWS = {"primary": (-90, 14), "strict": (-90, 0)}   # days from the anchor date, inclusive (frozen §15 item 4)
MARGINS = {"roc_auc": 0.02, "log_oe": math.log(1.10), "brier": 0.01}   # AUROC/log O:E frozen (§15 item 1); Brier new (§7)
SELECTION = {"criterion": "brier", "one_se_multiplier": 1.0, "max_auroc_loss": 0.02, "cal_slope_range": (0.8, 1.2),
             "abs_log_oe_max": math.log(1.10)}

# ------------------------------------------------------------------------------------------------ feature groups
FROZEN_FEATURES = ("age", "female", "education_years", "mmse", "adas13", "faq", "apoe4")
GROUPS = {
    "DEM": ("age", "female", "education_years"),
    "COG": ("mmse", "adas13"),
    "FUNC": ("faq",),
    "APOE": ("apoe4",),
    "COGX": ("ravlt_immediate", "ravlt_perc_forgetting", "ldeltotal", "trabscor", "catanimsc"),
    "CDR": ("cdrsb",),
    "MRI": ("mri_hippocampus", "mri_entorhinal", "mri_entorhinal_thickness", "mri_fusiform", "mri_midtemp",
            "mri_ventricles", "mri_amygdala"),
    "MRIFULL": (),                 # resolved at run time from experiments/simkhada_baseline/table_s1_mri_columns.txt
    "CSF": ("csf_log_ptau_abeta42", "csf_abeta42", "csf_ptau"),
    "PLASMA": ("plasma_ptau217", "plasma_ab42_ab40", "plasma_nfl", "plasma_gfap"),
    "AMY": ("amy_centiloid",),
    "SIMX": ("adas11", "cdglobal"),  # extra Simkhada cognitive inputs (with CDR and COG)
}
DIAGNOSIS_LINKED = ("mmse", "adas13", "faq", "cdrsb", "cdglobal", "adas11")   # decision log 127-128 (+ CDR, protocol §5)
INSTRUMENTS = {"age": "demographic", "female": "demographic", "education_years": "demographic", "mmse": "MMSE",
               "adas13": "ADAS-Cog", "adas11": "ADAS-Cog", "faq": "FAQ", "apoe4": "APOE genotype", "cdrsb": "CDR",
               "cdglobal": "CDR", **{f: "neuropsychological battery" for f in GROUPS["COGX"]},
               **{f: "structural MRI (FreeSurfer 7)" for f in GROUPS["MRI"]}, **{f: "CSF" for f in GROUPS["CSF"]},
               **{f: "plasma" for f in GROUPS["PLASMA"]}, **{f: "amyloid PET" for f in GROUPS["AMY"]}}

# ------------------------------------------------------------------------------------------------ feature sets
FEATURE_SETS = {                    # id -> groups (order fixes the column order)
    "FS01_DEM": ("DEM",), "FS02_COG": ("COG",), "FS03_FUNC": ("FUNC",), "FS04_DEM_COG": ("DEM", "COG"),
    "FS05_DEM_FUNC": ("DEM", "FUNC"), "FS06_COG_FUNC": ("COG", "FUNC"), "FS07_DEM_COG_FUNC": ("DEM", "COG", "FUNC"),
    "FS08_CORE7": ("DEM", "COG", "FUNC", "APOE"),                  # == frozen primary predictor set (asserted below)
    "FS08a_DEM_APOE": ("DEM", "APOE"),                             # == frozen S-noDx set (asserted below)
    "FS08b_TRANSPORT": ("DEM", "APOE", "+mmse", "+faq"),           # no ADAS-Cog: the NACC-portable reduced model
    "FS09_MRI": ("MRI",), "FS10_MRI_DEM": ("DEM", "MRI"), "FS11_MRI_COG": ("COG", "MRI"),
    "FS12_MRI_COG_FUNC": ("COG", "FUNC", "MRI"), "FS13_MRI_CORE7": ("DEM", "COG", "FUNC", "APOE", "MRI"),
    "FS13a_MRI_DEM_APOE": ("DEM", "APOE", "MRI"),                  # MRI without diagnosis-linked scales
    "FS14a_CORE7_CSF": ("DEM", "COG", "FUNC", "APOE", "CSF"), "FS14b_CORE7_PLASMA": ("DEM", "COG", "FUNC", "APOE", "PLASMA"),
    "FS14c_CORE7_AMY": ("DEM", "COG", "FUNC", "APOE", "AMY"),
    "FS15n_FULL_NOMRI": ("DEM", "COG", "FUNC", "APOE", "COGX", "CDR"),
    "FS15_FULL": ("DEM", "COG", "FUNC", "APOE", "COGX", "CDR", "MRI"),
    "FS16_CORE7_COGX": ("DEM", "COG", "FUNC", "APOE", "COGX"), "FS17_CORE7_CDR": ("DEM", "COG", "FUNC", "APOE", "CDR"),
    "FS18n_SIMKHADA_NOMRI": ("DEM", "COG", "SIMX", "CDR", "FUNC", "APOE"),
    "FS18_SIMKHADA_MAP": ("DEM", "COG", "SIMX", "CDR", "FUNC", "APOE", "MRIFULL"),
    "FS19_CORE7_MRIFULL": ("DEM", "COG", "FUNC", "APOE", "MRIFULL"),
}
LANDMARK_SETS = {
    "LM0_BASELINE_INFO": ("age_bl", "female", "education_years", "mmse_bl", "adas13_bl", "faq_bl", "apoe4"),
    "LM1_CURRENT": FROZEN_FEATURES,
    "LM2_CURRENT_HISTORY": FROZEN_FEATURES + ("mmse_slope_per_year", "adas13_slope_per_year", "faq_slope_per_year",
                                              "years_since_baseline"),
}
HISTORY_VARIABLES = ("mmse", "adas13", "faq")


def resolve(fs_id: str, mri_full: list | tuple = ()) -> list:
    """Ordered feature names of a feature set."""
    if fs_id in LANDMARK_SETS:
        return list(LANDMARK_SETS[fs_id])
    out = []
    for g in FEATURE_SETS[fs_id]:
        names = (g[1:],) if g.startswith("+") else (tuple(f"fs_{c}" for c in mri_full) if g == "MRIFULL" else GROUPS[g])
        out += [n for n in names if n not in out]
    return out


def groups_of(fs_id: str) -> list:
    """Modality groups of a feature set ('+feature' entries count under the group that owns the feature)."""
    if fs_id in LANDMARK_SETS:
        return ["LONGITUDINAL"]
    out = []
    for g in FEATURE_SETS[fs_id]:
        g = next(k for k, v in GROUPS.items() if g[1:] in v) if g.startswith("+") else g
        if g not in out:
            out.append(g)
    return out


assert resolve("FS08_CORE7") == frozen.PREDICTOR_SETS["primary"], "FS08 must equal the frozen primary predictor set"
assert resolve("FS08a_DEM_APOE") == frozen.PREDICTOR_SETS["S-noDx"], "FS08a must equal the frozen S-noDx set"

# ------------------------------------------------------------------------------------------------ populations
POPULATIONS = {     # all are REF-determinate baseline participants of the frozen TRAINING partition unless stated
    "P_ALL": "all REF-labelled development participants (the frozen A3 training rows)",
    "P_MRI": "P_ALL with a QC-acceptable FreeSurfer 7 scan in the predictor window",
    "P_CSF": "P_ALL with a CSF Elecsys assay in the predictor window",
    "P_PLASMA": "P_ALL with a plasma Fujirebio/Quanterix assay in the predictor window",
    "P_AMY": "P_ALL with an amyloid-PET centiloid value in the predictor window",
    "P_LM": "development participants whose landmark anchor (earliest eligible anchor at month >= 12) is REF-determinate",
}
INSTRUMENT_OF_POPULATION = {"P_MRI": "mri", "P_CSF": "csf", "P_PLASMA": "plasma", "P_AMY": "amy"}

# ------------------------------------------------------------------------------------------------ models
# rank: prespecified complexity order used by the selection rule (lower = simpler). grid: finite search grid (sampled without
# replacement with a seeded generator when larger than n_iter). tune_metric: inner-CV criterion.
MODELS = {
    "lr_l2_ref": {"rank": 1, "family": "penalised logistic regression (frozen reference: L2, C = 1, median + scaling)",
                  "fixed": {"C": 1.0}},
    "lr_enet": {"rank": 2, "family": "elastic-net logistic regression (saga)", "n_iter": 12, "tune_metric": "log_loss",
                "grid": {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}},
    "cox_l2": {"rank": 3, "family": "censoring-aware ridge Cox model (uses REF-indeterminate follow-up in training)",
               "n_iter": 4, "tune_metric": "log_loss", "grid": {"penalty": [0.01, 0.1, 1.0, 10.0]}},
    "lr_interact": {"rank": 4, "family": "ridge logistic regression with all pairwise interactions", "n_iter": 7,
                    "tune_metric": "log_loss", "grid": {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}},
    "late_fusion": {"rank": 5, "family": "decision-level fusion: per-modality L2 logistic models stacked by logistic regression",
                    "fixed": {"C": 1.0, "stack_C": 1.0}},
    "svm_rbf": {"rank": 6, "family": "RBF support vector machine (score-only) + Platt calibration on inner OOF scores",
                "n_iter": 12, "tune_metric": "roc_auc", "calibration": "platt",
                "grid": {"C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], "gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}},
    "xgb_ref": {"rank": 7, "family": "XGBoost, frozen secondary model (200 trees, depth 2, learning rate 0.05)",
                "fixed": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05}},
    "hist_gb": {"rank": 7, "family": "histogram gradient boosting (scikit-learn)", "n_iter": 12, "tune_metric": "log_loss",
                "grid": {"learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 200, 400],
                         "min_samples_leaf": [10, 20, 40], "l2_regularization": [0.0, 1.0]}},
    "xgb_tuned": {"rank": 8, "family": "XGBoost, tuned", "n_iter": 16, "tune_metric": "log_loss",
                  "grid": {"n_estimators": [100, 200, 400], "max_depth": [1, 2, 3, 4], "learning_rate": [0.02, 0.05, 0.1],
                           "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 5],
                           "reg_lambda": [1.0, 5.0]}},
    "random_forest": {"rank": 8, "family": "random forest", "n_iter": 10, "tune_metric": "log_loss",
                      "grid": {"n_estimators": [300], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10, 20],
                               "max_features": ["sqrt", 0.5]}},
    "extra_trees": {"rank": 8, "family": "extremely randomised trees", "n_iter": 10, "tune_metric": "log_loss",
                    "grid": {"n_estimators": [300], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10, 20],
                             "max_features": ["sqrt", 0.5]}},
    "xgb_tuned_platt": {"rank": 9, "family": "XGBoost, tuned, + Platt calibration on inner OOF predictions",
                        "derived_from": "xgb_tuned", "calibration": "platt"},
    "xgb_simkhada": {"rank": 10, "family": "Simkhada-style calibrated XGBoost: KNN imputation, variance/correlation filters, "
                     "SMOTENC, randomized search refit on balanced accuracy, grouped Platt calibration, Youden threshold",
                     "n_iter": 20, "inner_splits": 5, "search_space": "experiments.simkhada_baseline.baseline_lib."
                     "search_space('xgboost', 'smotenc')"},
    "mlp": {"rank": 11, "family": "small multilayer perceptron (no early stopping)", "n_iter": 8, "tune_metric": "log_loss",
            "grid": {"hidden_layer_sizes": [[8], [16], [32], [16, 8]], "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                     "learning_rate_init": [0.001, 0.003]}},
}
NOT_RUN_MODELS = {
    "lightgbm": "not installed in the approved environment (.venv-data); installing needs network access and a researcher "
                "decision. scikit-learn histogram gradient boosting and XGBoost cover gradient boosting.",
    "catboost": "not installed in the approved environment (.venv-data); same as lightgbm.",
    "longitudinal_neural_network": "not justified: about 430 labelled development participants, median of one pre-index "
                                   "assessment at baseline, sparse irregular repeated measures (PROTOCOL.md §6.4).",
}
REFERENCE = ("FS08_CORE7", "lr_l2_ref")
STAGE_A_MODELS = ("lr_l2_ref", "xgb_ref")
STAGE_B_MODELS = ("lr_enet", "cox_l2", "lr_interact", "late_fusion", "svm_rbf", "hist_gb", "xgb_tuned", "xgb_tuned_platt",
                  "random_forest", "extra_trees", "mlp")      # lr_l2_ref and xgb_ref on these sets come from stage A
STAGE_B_SETS = (("P_ALL", "FS08_CORE7"), ("P_ALL", "FS15n_FULL_NOMRI"), ("P_MRI", "FS13_MRI_CORE7"))
STAGE_A_SETS = {
    "P_ALL": ("FS01_DEM", "FS02_COG", "FS03_FUNC", "FS04_DEM_COG", "FS05_DEM_FUNC", "FS06_COG_FUNC", "FS07_DEM_COG_FUNC",
              "FS08_CORE7", "FS08a_DEM_APOE", "FS08b_TRANSPORT", "FS15n_FULL_NOMRI", "FS16_CORE7_COGX", "FS17_CORE7_CDR"),
    "P_MRI": ("FS01_DEM", "FS02_COG", "FS03_FUNC", "FS04_DEM_COG", "FS05_DEM_FUNC", "FS06_COG_FUNC", "FS07_DEM_COG_FUNC",
              "FS08_CORE7", "FS08a_DEM_APOE", "FS08b_TRANSPORT", "FS15n_FULL_NOMRI", "FS16_CORE7_COGX", "FS17_CORE7_CDR",
              "FS09_MRI", "FS10_MRI_DEM", "FS11_MRI_COG", "FS12_MRI_COG_FUNC", "FS13_MRI_CORE7", "FS13a_MRI_DEM_APOE",
              "FS15_FULL", "FS19_CORE7_MRIFULL"),
    "P_CSF": ("FS08_CORE7", "FS14a_CORE7_CSF"), "P_PLASMA": ("FS08_CORE7", "FS14b_CORE7_PLASMA"),
    "P_AMY": ("FS08_CORE7", "FS14c_CORE7_AMY"),
}
HIGH_DIM_SETS = ("FS19_CORE7_MRIFULL",)          # also fitted with lr_enet in stage A
SIMKHADA_SETS = (("P_ALL", "FS18n_SIMKHADA_NOMRI"), ("P_MRI", "FS18_SIMKHADA_MAP"))
IMPUTATION_VARIANTS = ("iterative", "median_indicator", "complete_case")   # stage D on (P_ALL, FS08_CORE7, lr_l2_ref)
LANDMARK_MODELS = ("lr_l2_ref", "xgb_ref", "lr_enet")


def candidate_id(stage, population, fs, model, imputation="median", timing="primary") -> str:
    return f"{stage}|{population}|{fs}|{model}|{imputation}|{timing}"


def parse_candidate(cid: str) -> dict:
    stage, population, fs, model, imputation, timing = cid.split("|")
    return dict(stage=stage, population=population, feature_set=fs, model=model, imputation=imputation, timing=timing)


def experiment_matrix(populations: set) -> list[dict]:
    """Structured matrix (never a full Cartesian product). `populations` = populations that passed the feasibility gate.
    Stage E (strict-timing sensitivity of the selected model) and F (locked held-out evaluation) are added after selection."""
    out, seen = [], set()

    def add(stage, pop, fs, model, imputation="median", timing="primary"):
        if pop not in populations:
            return
        cid = candidate_id(stage, pop, fs, model, imputation, timing)
        if cid not in seen:
            seen.add(cid); out.append(parse_candidate(cid) | {"id": cid})
    for pop, sets in STAGE_A_SETS.items():
        for fs in sets:
            for m in STAGE_A_MODELS + (("lr_enet",) if fs in HIGH_DIM_SETS else ()):
                add("A", pop, fs, m)
    for pop, fs in STAGE_B_SETS:
        for m in STAGE_B_MODELS:
            if m == "late_fusion" and len(groups_of(fs)) < 2:
                continue
            if m == "cox_l2" and pop != "P_ALL":
                continue
            add("B", pop, fs, m)
    for pop, fs in SIMKHADA_SETS:
        add("B", pop, fs, "xgb_simkhada")
    for v in IMPUTATION_VARIANTS:
        add("D", "P_ALL", "FS08_CORE7", "lr_l2_ref", imputation=v)
    for fs in LANDMARK_SETS:
        for m in LANDMARK_MODELS:
            add("C", "P_LM", fs, m)
    return out


def contrast_plan(candidates: list[str]) -> list[dict]:
    """Prespecified paired contrasts among the candidates that exist: (candidate, comparator, family). Evaluated on the
    participants that have out-of-fold predictions from both."""
    have = set(candidates)
    out = []

    def add(a, b, family):
        if a in have and b in have and a != b:
            out.append({"candidate": a, "comparator": b, "family": family})
    for c in candidates:
        p = parse_candidate(c)
        if p["stage"] == "A" and p["feature_set"] != "FS08_CORE7":
            add(c, candidate_id("A", p["population"], "FS08_CORE7", p["model"]), "modality_vs_core7")
    mri_pairs = (("FS13_MRI_CORE7", "FS08_CORE7"), ("FS13a_MRI_DEM_APOE", "FS08a_DEM_APOE"), ("FS10_MRI_DEM", "FS01_DEM"),
                 ("FS11_MRI_COG", "FS02_COG"), ("FS12_MRI_COG_FUNC", "FS06_COG_FUNC"), ("FS15_FULL", "FS15n_FULL_NOMRI"))
    for a, b in mri_pairs:
        for m in STAGE_A_MODELS:
            add(candidate_id("A", "P_MRI", a, m), candidate_id("A", "P_MRI", b, m), "mri_increment")
    add(candidate_id("A", "P_MRI", "FS19_CORE7_MRIFULL", "lr_enet"), candidate_id("A", "P_MRI", "FS08_CORE7", "lr_l2_ref"),
        "mri_increment")
    for c in candidates:
        p = parse_candidate(c)
        if p["stage"] == "B" and p["model"] != "lr_l2_ref":
            ref_stage = "A" if candidate_id("A", p["population"], p["feature_set"], "lr_l2_ref") in have else "B"
            add(c, candidate_id(ref_stage, p["population"], p["feature_set"], "lr_l2_ref"), "model_vs_reference_lr")
    for pop, fs in STAGE_B_SETS + SIMKHADA_SETS:
        add(candidate_id("B", pop, fs, "xgb_tuned_platt"), candidate_id("B", pop, fs, "xgb_tuned"), "calibration_effect")
        add(candidate_id("B", pop, fs, "lr_interact"), candidate_id("B", pop, fs, "lr_enet"), "interaction_effect")
    for pop, fs in SIMKHADA_SETS:
        add(candidate_id("B", pop, fs, "xgb_simkhada"), candidate_id("A", pop, "FS08_CORE7", "lr_l2_ref"),
            "simkhada_vs_reference")
    for v in IMPUTATION_VARIANTS:
        add(candidate_id("D", "P_ALL", "FS08_CORE7", "lr_l2_ref", imputation=v),
            candidate_id("A", "P_ALL", "FS08_CORE7", "lr_l2_ref"), "imputation")
    for m in LANDMARK_MODELS:
        add(candidate_id("C", "P_LM", "LM1_CURRENT", m), candidate_id("C", "P_LM", "LM0_BASELINE_INFO", m), "longitudinal")
        add(candidate_id("C", "P_LM", "LM2_CURRENT_HISTORY", m), candidate_id("C", "P_LM", "LM1_CURRENT", m), "longitudinal")
    return out


def grid_configs(model_id: str) -> list[dict]:
    """The prespecified configurations evaluated for one tuned model, identical in every outer fold: the full grid when it is
    not larger than n_iter, else n_iter configurations drawn without replacement by a generator seeded from SEED and the
    model id. Every configuration is recorded in the run outputs."""
    import numpy as np
    spec = MODELS[model_id]
    keys = sorted(spec["grid"])
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(spec["grid"][k] for k in keys))]
    if len(combos) <= spec["n_iter"]:
        return combos
    seed = (SEED + int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)) % (2 ** 32)
    pick = np.random.default_rng(seed).choice(len(combos), size=spec["n_iter"], replace=False)
    return [combos[i] for i in sorted(pick)]


def plan_dict(mri_full: list | tuple = ()) -> dict:
    """Everything that defines the scientific plan, as JSON-serialisable data (input to config_sha256)."""
    return {"seed": SEED, "outer": [OUTER_SPLITS, OUTER_REPEATS], "inner": INNER_SPLITS, "boot": [BOOT_B, ALPHA],
            "k_suppress": K_SUPPRESS, "gate": [MIN_EVENTS, MIN_NONEVENTS, FULL_FROM_EVENTS],
            "max_missing": MAX_MISSING_FRACTION, "interaction_max_base": INTERACTION_MAX_BASE,
            "landmark_min_month": LANDMARK_MIN_MONTH, "windows": WINDOWS, "margins": MARGINS, "selection": SELECTION,
            "groups": GROUPS, "feature_sets": FEATURE_SETS, "landmark_sets": LANDMARK_SETS, "history": HISTORY_VARIABLES,
            "populations": POPULATIONS, "models": MODELS, "not_run": NOT_RUN_MODELS, "reference": REFERENCE,
            "stage_a": STAGE_A_SETS, "stage_a_models": STAGE_A_MODELS, "stage_b": [STAGE_B_SETS, STAGE_B_MODELS],
            "high_dim": HIGH_DIM_SETS, "simkhada": SIMKHADA_SETS, "imputation": IMPUTATION_VARIANTS,
            "landmark_models": LANDMARK_MODELS, "diagnosis_linked": DIAGNOSIS_LINKED, "mri_full": list(mri_full)}


def config_sha256(mri_full: list | tuple = ()) -> str:
    return hashlib.sha256(json.dumps(plan_dict(mri_full), sort_keys=True, default=str).encode()).hexdigest()
