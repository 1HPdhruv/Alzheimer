"""Experiment configuration: defaults, YAML loading and validation.

Validation happens before any data is generated. It rejects unknown keys (so a config cannot smuggle in an input path),
unregistered or planned components, incompatible options and output locations outside `results/`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import PurePosixPath

import yaml

from yoad.benchmark.data import ADAPTERS
from yoad.benchmark.metrics import METRICS
from yoad.benchmark.models import REGISTRY, ModelNotAvailable, get_model
from yoad.benchmark.pipeline import IMPUTER_FACTORIES, NUMERIC_IMPUTATION, PLANNED_IMPUTATION, SAMPLERS, STEP_HOOKS


class ConfigError(ValueError):
    pass


TASKS_IMPLEMENTED = {"binary_classification"}
TASKS_PLANNED = {"survival", "longitudinal_classification", "multiclass_classification"}

DEFAULTS = {
    "experiment_id": None,
    "description": "",
    "task": "binary_classification",
    "seed": 0,
    "dataset": {"adapter": "synthetic", "params": {}},
    "features": {"groups": None},                 # None = all groups of the dataset
    "fusion": {"type": "early", "inner_cv_splits": 3},   # early = one pipeline on concatenated groups; late = see runner
    "preprocessing": {"numeric_imputation": "median", "scaling": "standard", "low_variance_threshold": None,
                      "correlation_threshold": None, "extra_steps": []},
    "feature_selection": {"method": "none", "k": None, "l1_C": None, "n_components": None},
    "sampling": {"method": "none"},
    "model": {"id": "logistic_l2", "params": {}, "search_space": {}, "search_metric": "log_loss"},
    "validation": {"outer": {"type": "group_holdout", "test_fraction": 0.25, "n_splits": None},
                   "validation_fraction": 0.0, "inner_cv_splits": 3, "stratify": True, "unit": "participant"},
    "calibration": {"method": "none"},
    "threshold": {"method": "fixed", "value": 0.5},
    "metrics": ["roc_auc", "average_precision", "brier", "log_loss", "balanced_accuracy", "sensitivity", "specificity",
                "f1_macro"],
    "output": {"root": "results/benchmark"},
}
SEARCH_METRICS = {"log_loss", "roc_auc", "brier", "average_precision"}   # probability metrics usable for model selection


def _merge(base: dict, over: dict, path="") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"unknown configuration key '{path}{k}'")
        if isinstance(base[k], dict) and k not in ("params", "search_space"):
            if not isinstance(v, dict):
                raise ConfigError(f"'{path}{k}' must be a mapping")
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def load_config(path_or_dict) -> dict:
    raw = path_or_dict
    if not isinstance(raw, dict):
        with open(path_or_dict) as f:            # the only file read by the framework: the YAML config itself
            raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a mapping")
    cfg = _merge(DEFAULTS, raw)
    validate(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def _req(cond, msg):
    if not cond:
        raise ConfigError(msg)


def validate(cfg: dict) -> None:
    _req(isinstance(cfg["experiment_id"], str) and cfg["experiment_id"].replace("_", "").replace("-", "").isalnum(),
         "experiment_id must be a non-empty string of letters, digits, '_' or '-'")
    t = cfg["task"]
    _req(t in TASKS_IMPLEMENTED | TASKS_PLANNED, f"unknown task {t!r}")
    _req(t in TASKS_IMPLEMENTED, f"task {t!r} is planned but not implemented in Stage 1")
    _req(isinstance(cfg["seed"], int) and not isinstance(cfg["seed"], bool), "seed must be an integer")

    ds = cfg["dataset"]
    _req(ds["adapter"] in ADAPTERS, f"unknown dataset adapter {ds['adapter']!r}; registered: {sorted(ADAPTERS)}")
    ad = ADAPTERS[ds["adapter"]]
    _req(isinstance(ds["params"], dict), "dataset.params must be a mapping")
    bad = set(ds["params"]) - ad.allowed_params
    _req(not bad, f"dataset.params {sorted(bad)} not accepted by adapter {ad.name!r}; allowed {sorted(ad.allowed_params)}")
    groups = cfg["features"]["groups"]
    if groups is not None:
        _req(isinstance(groups, list) and groups and len(set(groups)) == len(groups),
             "features.groups must be a non-empty list without duplicates, or null")
        unknown = set(groups) - set(ad.feature_groups)
        _req(not unknown, f"features.groups {sorted(unknown)} unknown for adapter {ad.name!r}; available {list(ad.feature_groups)}")

    fu = cfg["fusion"]
    _req(fu["type"] in {"early", "late", "learned"}, f"unknown fusion.type {fu['type']!r}")
    _req(fu["type"] != "learned", "fusion.type 'learned' is planned but not implemented")
    if fu["type"] == "late":
        n_groups = len(groups) if groups is not None else len(ad.feature_groups)
        _req(n_groups >= 2, "late fusion needs at least two feature groups")
        _req(isinstance(fu["inner_cv_splits"], int) and fu["inner_cv_splits"] >= 2, "fusion.inner_cv_splits must be an integer >= 2")
        _req(not cfg["model"]["search_space"], "late fusion with a hyperparameter search is not implemented in Stage 1; use model.params")

    pre = cfg["preprocessing"]
    _req(pre["numeric_imputation"] in set(NUMERIC_IMPUTATION) | set(IMPUTER_FACTORIES) | PLANNED_IMPUTATION,
         f"unknown numeric_imputation {pre['numeric_imputation']!r}")
    _req(pre["numeric_imputation"] not in PLANNED_IMPUTATION,
         f"numeric_imputation {pre['numeric_imputation']!r} is planned but not implemented")
    _req(pre["scaling"] in {"standard", "none"}, "preprocessing.scaling must be 'standard' or 'none'")
    lv = pre["low_variance_threshold"]
    _req(lv is None or (isinstance(lv, (int, float)) and lv >= 0), "low_variance_threshold must be null or >= 0")
    ct = pre["correlation_threshold"]
    _req(ct is None or (isinstance(ct, (int, float)) and 0 < ct < 1), "correlation_threshold must be null or in (0, 1)")
    for h in pre["extra_steps"]:
        _req(h in STEP_HOOKS, f"preprocessing.extra_steps {h!r} is not a registered step hook")

    fs = cfg["feature_selection"]
    _req(fs["method"] in {"none", "anova", "mutual_info", "l1", "pca", "stability", "rfe"},
         f"unknown feature_selection.method {fs['method']!r}")
    _req(fs["method"] not in {"stability", "rfe"}, f"feature_selection.method {fs['method']!r} is planned but not implemented")
    if fs["method"] in {"anova", "mutual_info"}:
        _req(isinstance(fs["k"], int) and fs["k"] >= 1, "feature_selection.k must be a positive integer for anova/mutual_info")
    else:
        _req(fs["k"] is None, "feature_selection.k is only valid with anova or mutual_info")
    if fs["method"] == "l1":
        _req(isinstance(fs["l1_C"], (int, float)) and fs["l1_C"] > 0, "feature_selection.l1_C must be > 0 for l1 selection")
    else:
        _req(fs["l1_C"] is None, "feature_selection.l1_C is only valid with method l1")
    if fs["method"] == "pca":
        nc = fs["n_components"]
        _req((isinstance(nc, int) and nc >= 1) or (isinstance(nc, float) and 0 < nc < 1),
             "feature_selection.n_components must be a positive integer or a variance fraction in (0, 1)")
    else:
        _req(fs["n_components"] is None, "feature_selection.n_components is only valid with method pca")

    sm = cfg["sampling"]["method"]
    _req(sm in {"none", "class_weight"} | set(SAMPLERS), f"unknown sampling.method {sm!r}")
    _req(SAMPLERS.get(sm, True) is not None, f"sampling.method {sm!r} is planned but not implemented (needs imbalanced-learn)")

    m = cfg["model"]
    try:
        spec = get_model(m["id"], t)
    except ModelNotAvailable as e:
        raise ConfigError(str(e)) from None
    for key in ("params", "search_space"):
        _req(isinstance(m[key], dict), f"model.{key} must be a mapping")
        bad = set(m[key]) - spec.allowed_params
        _req(not bad, f"model.{key} {sorted(bad)} not allowed for {m['id']!r}; allowed {sorted(spec.allowed_params)}")
    _req(not set(m["params"]) & set(m["search_space"]), "a parameter cannot be both fixed (params) and searched (search_space)")
    for k, v in m["search_space"].items():
        _req(isinstance(v, list) and len(v) >= 1, f"model.search_space.{k} must be a non-empty list")
    _req(m["search_metric"] in SEARCH_METRICS, f"model.search_metric must be one of {sorted(SEARCH_METRICS)}")
    if sm == "class_weight":
        _req(spec.supports_class_weight, f"model {m['id']!r} does not support class_weight sampling")
    if not spec.probabilistic:
        _req(cfg["calibration"]["method"] != "none",
             f"model {m['id']!r} outputs scores, not probabilities; set calibration.method (fitted on the validation partition)")
        _req(m["search_metric"] in {"roc_auc", "average_precision"},
             f"model {m['id']!r} outputs scores; use a ranking search_metric (roc_auc or average_precision)")

    v = cfg["validation"]
    o = v["outer"]
    _req(v["unit"] == "participant", "validation.unit must be 'participant' (observation-level splits are not supported)")
    _req(o["type"] in {"group_holdout", "group_kfold"}, f"validation.outer.type must be group_holdout or group_kfold; got {o['type']!r}")
    if o["type"] == "group_holdout":
        tf = o["test_fraction"]
        _req(isinstance(tf, (int, float)) and 0 < tf < 1, "validation.outer.test_fraction must be in (0, 1)")
        _req(o["n_splits"] is None, "validation.outer.n_splits is only valid with group_kfold")
    else:
        _req(isinstance(o["n_splits"], int) and o["n_splits"] >= 2, "validation.outer.n_splits must be an integer >= 2")
    vf = v["validation_fraction"]
    _req(isinstance(vf, (int, float)) and 0 <= vf < 0.5, "validation.validation_fraction must be in [0, 0.5)")
    if o["type"] == "group_holdout":
        _req(o["test_fraction"] + vf < 0.8, "test_fraction + validation_fraction must leave at least 20% for fitting")
    ic = v["inner_cv_splits"]
    _req(isinstance(ic, int) and ic >= 2, "validation.inner_cv_splits must be an integer >= 2")
    _req(isinstance(v["stratify"], bool), "validation.stratify must be true or false")

    cal = cfg["calibration"]["method"]
    _req(cal in {"none", "sigmoid", "isotonic"}, f"unknown calibration.method {cal!r}")
    th = cfg["threshold"]
    _req(th["method"] in {"fixed", "youden", "target_sensitivity"}, f"unknown threshold.method {th['method']!r}")
    if th["method"] in {"fixed", "target_sensitivity"}:
        _req(isinstance(th["value"], (int, float)) and 0 < th["value"] < 1, "threshold.value must be in (0, 1)")
    if cal != "none" or th["method"] != "fixed":
        _req(vf > 0, "calibration or data-driven threshold selection needs validation.validation_fraction > 0 "
                     "(they are never fitted on the evaluation partition)")

    mets = cfg["metrics"]
    _req(isinstance(mets, list) and mets and len(set(mets)) == len(mets), "metrics must be a non-empty list without duplicates")
    for name in mets:
        _req(name in METRICS, f"unknown metric {name!r}; available {sorted(METRICS)}")
        _req(t in METRICS[name].tasks, f"metric {name!r} not defined for task {t!r}")

    root = cfg["output"]["root"]
    p = PurePosixPath(str(root))
    _req(isinstance(root, str) and not p.is_absolute() and ".." not in p.parts and p.parts[:1] == ("results",),
         "output.root must be a relative path inside 'results/'")
