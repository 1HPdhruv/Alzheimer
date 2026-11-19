"""Prespecified plan of temporal multimodal v1 (PROTOCOL.md). Pure data and pure functions; no file or data access.

Everything that defines the analysis enters config_sha256(); the guarded runner refuses to start when FREEZE.json disagrees.
Feature names, seeds, folds and gates come from the frozen benchmark registry; nothing is redefined here.
"""
from __future__ import annotations

import hashlib
import json
import math

from yoad.multimodal import registry as R

VERSION = "temporal-multimodal-v1"

# ------------------------------------------------------------------------------------------------ validation and uncertainty
SEED = R.SEED                                   # frozen seed; outer folds = multimodal_benchmark.cohort.outer_folds (5 x 3)
REPEATS, SPLITS = R.OUTER_REPEATS, R.OUTER_SPLITS
INNER_SPLITS = 3                                # StratifiedGroupKFold over anchors, grouped by participant
BOOT_B, ALPHA = R.BOOT_B, R.ALPHA
K_SUPPRESS = R.K_SUPPRESS
GATE_EVENTS, GATE_NONEVENTS = R.MIN_EVENTS, R.MIN_NONEVENTS     # event PARTICIPANTS / non-event participants (unchanged 50/50)
MIN_FOLD_CLASS = 2                              # every outer test fold needs >= 2 event and >= 2 non-event anchors
NEURAL_SEEDS = (0, 1, 2)                        # final refit inside every outer fold: 3 seeds, predictions averaged
THRESHOLD_RULE = ("one rule for every candidate: Youden index on out-of-fold predictions over the SAME 3 grouped inner splits "
                  "of the outer-training anchors (chosen configuration); never evaluation outcomes")
CALIBRATION_RULE = ("every candidate's native output is reported; in addition every candidate is Platt-recalibrated on its "
                    "inner out-of-fold predictions (training anchors only) and reported separately; calibration eligibility "
                    "uses the recalibrated output for every candidate")


def inner_seed(r: int, k: int) -> int:
    """Frozen engine.build_tasks convention."""
    return SEED + 100 * r + k + 1


def torch_seed(r: int, k: int, s: int, stage: str) -> int:
    return SEED + 10_000 * (r + 1) + 100 * k + s + (0 if stage == "final" else 50)


# ------------------------------------------------------------------------------------------------ prediction time
# Only information dated ON OR BEFORE the MCI anchor date is used (index time = anchor date). The frozen +14-day assessment
# window of the earlier analyses (window (-90, +14)) is EXCLUDED from this primary predictive experiment; the 24-month
# endpoint is unchanged (REF, horizon = anchor date + 24 calendar months).
POST_DAYS = 0
TIMING = "strict"                                # frozen registry window name: (-90, 0) for the anchor-visit snapshot
WINDOW = R.WINDOWS[TIMING]

# ------------------------------------------------------------------------------------------------ feature blocks (7)
TV_BLOCKS = {                                    # time-varying: dated assessment streams
    "COGFUNC": ("mmse", "adas13", "faq", "cdrsb") + tuple(R.GROUPS["COGX"]),
    "MRI": tuple(R.GROUPS["MRI"]),
    "CSF": tuple(R.GROUPS["CSF"]),
    "PLASMA": tuple(R.GROUPS["PLASMA"]),
    "AMY": tuple(R.GROUPS["AMY"]),
}
STATIC_BLOCKS = {"DEM": ("age", "female", "education_years"), "APOE": ("apoe4",)}
BLOCKS = ("DEM", "APOE", "COGFUNC", "MRI", "CSF", "PLASMA", "AMY")
BIOMARKER_BLOCKS = ("MRI", "CSF", "PLASMA", "AMY")
WIDE_PREFIX = {"MRI": "mri", "CSF": "csf", "PLASMA": "plasma", "AMY": "amy"}
LONG_VARIABLES = ("mmse", "adas13", "faq", "cdrsb") + tuple(R.GROUPS["COGX"])
LOG_FEATURES = ("csf_abeta42", "csf_ptau", "plasma_ptau217", "plasma_nfl", "plasma_gfap", "trabscor", "mri_ventricles")
CLIP_Z = 6.0                                     # standardised values clipped to +/- 6 (training statistics only)
MAX_PER_MODALITY = 12                            # most recent dated assessments kept per modality stream
MAX_DATES = 24                                   # most recent observation dates in the merged view (+ anchor token)
HISTORY_YEARS = 15.0                             # assessments older than this before the anchor are ignored

# ------------------------------------------------------------------------------------------------ training-evidence rules
MIN_FEATURE_TRAIN_OBS = 10       # a feature is ACTIVE in a fold only with >= 10 finite observed training values (else masked
                                 # out for every anchor of that fold, training and evaluation alike)
MIN_BRANCH_TRAIN_PARTICIPANTS = 20   # a modality branch is ENABLED in a fold only if >= 20 training participants have it
                                     # observed (technical rule; claims need the 50/50 branch eligibility of §9)

# ------------------------------------------------------------------------------------------------ endpoints
PRIMARY_ENDPOINT = "any"                         # REF any-cause dementia within 24 calendar months (frozen)
SECONDARY_ENDPOINT = "ad"                        # REF AD-type dementia (corrected etiology mapper; never mixed -> AD)

# ------------------------------------------------------------------------------------------------ candidates
SIMKHADA_FEATURES = ("mmse", "cdglobal", "cdrsb", "adas11", "adas13", "faq", "age", "female", "education_years", "apoe4")
SIMKHADA_SEARCH = {"n_iter": 20, "inner_splits": 5}   # = benchmark xgb_simkhada (published 250 / 1,000 draws: compute)
# Pinned reference code (commit c67358d, mci_progression_ml/framework/search_config.py, final/method.py): SelectKBest step
# "select" placed AFTER SMOTENC and before the classifier, used whenever FreeSurfer features are included.
SIMKHADA_SELECTOR = {"k": (20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150), "score_func": ("f_classif", "mutual_info_classif"),
                     "default_k": 100, "mi_random_state": 11}
NEURAL_GRID = ({"hidden": 16}, {"hidden": 32})        # the only tuned hyperparameter; inner grouped CV, weighted log loss
NEURAL_FIXED = {"dropout": 0.2, "weight_decay": 1e-3, "lr": 3e-3, "batch_size": 64, "max_epochs": 150, "patience": 20,
                "heads": 2, "layers": 1, "modality_dropout": 0.2}
CANDIDATES = {   # id -> (family, description, selection role)
    "LR_REF": ("baseline", "frozen CORE7 L2 logistic (C = 1), anchor-visit snapshot (strict window)", "comparator"),
    "A0_XGB_INCUMBENT": ("baseline", "frozen XGBoost 200/2/0.05 on FS16_CORE7_COGX, anchor-visit snapshot (benchmark "
                         "incumbent)", "comparator"),
    "A_SIMKHADA": ("baseline", "adapted Simkhada-compatible calibrated XGBoost: 10 clinical + 323 Table S1 FreeSurfer "
                   "features, scaled KNN, filters, SMOTENC, SelectKBest (after SMOTENC), randomized search refit on balanced "
                   "accuracy, grouped Platt", "matched_baseline"),
    "A1_XGB_MATCHED": ("baseline", "frozen XGBoost 200/2/0.05 on the neural models' inputs as last observations (values, "
                       "staleness, availability)", "comparator"),
    "B_LSTM": ("neural", "LSTM over dated observation tokens (modality-aware projections, masks, elapsed time, gaps) + "
               "anchor token + static pathway", "candidate"),
    "C_TRANSFORMER": ("neural", "Transformer encoder over dated observation tokens with time embedding + static pathway",
                      "candidate"),
    "D_FUSION": ("neural", "missing-modality-aware gated fusion of last observations with staleness and modality dropout",
                 "candidate"),
    "E_COMBINED": ("neural", "Transformer over per-modality dated assessment tokens (own elapsed times, within-modality gaps): "
                   "temporal and cross-modal attention, modality embeddings, modality dropout, static token", "candidate"),
    "E_SNAP": ("neural", "E_COMBINED restricted to each modality's most recent assessment (history ablation)", "ablation"),
    "AVAIL": ("diagnostic", "logistic regression on modality availability and history length only", "diagnostic"),
}
NEURAL = tuple(k for k, v in CANDIDATES.items() if v[0] == "neural")
SELECTABLE_NEURAL = tuple(k for k, v in CANDIDATES.items() if v[2] == "candidate")
MATCHED_BASELINE = "A_SIMKHADA"
COMPLEXITY_ORDER = ("LR_REF", "A0_XGB_INCUMBENT", "A1_XGB_MATCHED", "A_SIMKHADA", "D_FUSION", "B_LSTM", "C_TRANSFORMER",
                    "E_COMBINED")             # simpler first; used by the one-SE rule for the overall best supported model

# ------------------------------------------------------------------------------------------------ modality ablations
# Refitted on identical participants and folds (replaces the withdrawn permutation diagnostic): drop_<block> removes the block
# from training AND evaluation; values_masked_biomarkers keeps biomarker masks/times but removes their values (shortcut test).
ABLATIONS = tuple(f"drop_{b}" for b in BLOCKS) + ("values_masked_biomarkers",)
ABLATION_MODELS = ("final_neural", "A1_XGB_MATCHED")

# ------------------------------------------------------------------------------------------------ evaluation and selection
METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "log_oe", "calibration_intercept", "calibration_slope")
THRESHOLD_METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv")
CONTRAST_METRICS = ("balanced_accuracy", "accuracy", "roc_auc", "average_precision", "brier", "log_loss", "calibration_slope")
WEIGHTED_METRICS = ("balanced_accuracy", "accuracy", "sensitivity", "specificity", "roc_auc", "brier", "log_loss",
                    "calibration_intercept", "calibration_slope")
EVAL_SETS = ("all_anchors", "baseline_anchor", "followup_anchors")
PRIMARY_EVAL = "all_anchors"
ESTIMAND = ("PRIMARY: anchor-weighted (every REF-determinate eligible MCI anchor weight 1); SENSITIVITY: participant-weighted "
            "(each participant's anchors sum to 1); uncertainty: participant-cluster bootstrap for both")
SELECTION = {
    "primary_metric": "balanced_accuracy (anchor-weighted, all anchors)",
    "improvement": "paired 95% participant-bootstrap CI of (candidate - A_SIMKHADA) balanced accuracy wholly above 0",
    "auroc_noninferiority_margin": 0.02,         # lower CI of (candidate - A_SIMKHADA) AUROC must exceed -0.02
    "calibration_output": "recalibrated",        # inner-OOF Platt output for every candidate
    "cal_slope_range": tuple(R.SELECTION["cal_slope_range"]),
    "abs_log_oe_max": R.SELECTION["abs_log_oe_max"],
    "max_seed_sd_balanced_accuracy": 0.02,
    "max_fold_sd_balanced_accuracy": 0.06,
    "tie": "candidates within one paired SE of the best balanced accuracy -> fewest trainable parameters",
}
SHORTCUT_AUROC = 0.55            # availability-only AUROC lower CI above this -> availability is predictive (flag)

# ------------------------------------------------------------------------------------------------ Simkhada references
SIMKHADA_PUBLISHED = {"samples": 2008, "participants": 828, "pMCI": 702, "sMCI": 1306, "cv_balanced_accuracy": 0.8128,
                      "cv_balanced_accuracy_sd": 0.0256, "test_balanced_accuracy": 0.8199, "test_ba_ci": (0.7822, 0.8551),
                      "test_roc_auc": 0.886, "test_pr_auc": 0.791, "test_brier": 0.131}
AGREEMENT_COUNT_TOLERANCE = 0.10    # decision log 105 (proposed; PI-confirmed for row B only, entry 122)


def plan_dict() -> dict:
    return {"version": VERSION, "seed": SEED, "repeats": REPEATS, "splits": SPLITS, "inner_splits": INNER_SPLITS,
            "boot": [BOOT_B, ALPHA], "k_suppress": K_SUPPRESS, "gate": [GATE_EVENTS, GATE_NONEVENTS],
            "min_fold_class": MIN_FOLD_CLASS, "neural_seeds": list(NEURAL_SEEDS), "threshold_rule": THRESHOLD_RULE,
            "calibration_rule": CALIBRATION_RULE, "post_days": POST_DAYS, "timing": TIMING, "window": list(WINDOW),
            "tv_blocks": {k: list(v) for k, v in TV_BLOCKS.items()}, "static_blocks": {k: list(v) for k, v in STATIC_BLOCKS.items()},
            "log_features": list(LOG_FEATURES), "clip_z": CLIP_Z, "max_per_modality": MAX_PER_MODALITY,
            "max_dates": MAX_DATES, "history_years": HISTORY_YEARS, "min_feature_train_obs": MIN_FEATURE_TRAIN_OBS,
            "min_branch_train_participants": MIN_BRANCH_TRAIN_PARTICIPANTS,
            "endpoints": [PRIMARY_ENDPOINT, SECONDARY_ENDPOINT], "simkhada_features": list(SIMKHADA_FEATURES),
            "simkhada_search": SIMKHADA_SEARCH, "simkhada_selector": SIMKHADA_SELECTOR, "neural_grid": list(NEURAL_GRID),
            "neural_fixed": NEURAL_FIXED, "candidates": {k: list(v) for k, v in CANDIDATES.items()},
            "matched_baseline": MATCHED_BASELINE, "complexity_order": list(COMPLEXITY_ORDER), "ablations": list(ABLATIONS),
            "ablation_models": list(ABLATION_MODELS), "metrics": list(METRICS), "threshold_metrics": list(THRESHOLD_METRICS),
            "contrast_metrics": list(CONTRAST_METRICS), "weighted_metrics": list(WEIGHTED_METRICS),
            "eval_sets": list(EVAL_SETS), "primary_eval": PRIMARY_EVAL, "estimand": ESTIMAND, "selection": SELECTION,
            "shortcut_auroc": SHORTCUT_AUROC, "simkhada_published": SIMKHADA_PUBLISHED,
            "agreement_count_tolerance": AGREEMENT_COUNT_TOLERANCE}


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(plan_dict(), sort_keys=True, default=str).encode()).hexdigest()


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
