"""Documented, deterministic predictor scoring for the Tier-1 set (synthetic-tested; used by the local pipeline).

Sources of the documented codes (public documentation only; no participant data):
  ADNI  - ADNIMERGE2 package man pages FAQ.Rd / CDR.Rd / PTDEMOG.Rd (data dictionary text);
  NACC  - public UDS data dictionary (uds4-rdd.csv): B7 items, CDRSUM, EDUC, NACCAGE, NACCSEX.
Nothing here is a validated crosswalk between instruments; the FAQ policy below is a *prespecified deterministic recode that the
owner must approve* (`faq.policy_approved_by_owner`), and it is applied identically to both cohorts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAQ_ITEMS_ADNI = ["FAQFINAN", "FAQFORM", "FAQSHOP", "FAQGAME", "FAQBEVG", "FAQMEAL", "FAQEVENT", "FAQTV", "FAQREM", "FAQTRAVL"]
FAQ_ITEMS_NACC = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]

# ADNI raw item codes -> ADNI's own item scores are 0,0,1,1,2,3 (FAQ.Rd). 'Never did' codes (1,2) carry an ADNI score (0 / 1) that
# NACC cannot reproduce (NACC has only '8 = Not applicable (e.g., never did)'), so the harmonised policy sets them to missing.
ADNI_CODE_TO_ADNI_SCORE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
ADNI_CODE_TO_HARMONISED = {0: 0.0, 1: np.nan, 2: np.nan, 3: 1.0, 4: 2.0, 5: 3.0}
NACC_CODE_TO_HARMONISED = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 8: np.nan, 9: np.nan, -4: np.nan}

FAQ_POLICIES = ("harmonised_prorated", "adni_provided_total")


class ScoringError(ValueError):
    """Unresolved or invalid scoring definition (message never contains participant values)."""


def _score_items(items: pd.DataFrame, code_map: dict, label: str) -> pd.DataFrame:
    vals = items.apply(pd.to_numeric, errors="coerce")
    known = set(code_map) | {np.nan}
    unknown = ~vals.isin(list(code_map)) & vals.notna()
    if unknown.any().any():
        cols = [c for c in vals.columns if unknown[c].any()]
        raise ScoringError(f"{label}: item codes outside the documented set in columns {cols}; scoring definition unresolved")
    return vals.apply(lambda s: s.map(code_map))


def faq_total_harmonised(items: pd.DataFrame, cohort: str, min_valid_items: int = 8) -> pd.Series:
    """Prorated total on the 0-30 scale: mean of valid harmonised item scores x 10, requiring >= min_valid_items valid items."""
    if items.shape[1] != 10:
        raise ScoringError("FAQ requires exactly 10 item columns")
    cmap = {"ADNI": ADNI_CODE_TO_HARMONISED, "NACC": NACC_CODE_TO_HARMONISED}.get(cohort)
    if cmap is None:
        raise ScoringError(f"unknown cohort {cohort!r} for FAQ scoring")
    sc = _score_items(items, cmap, f"FAQ[{cohort}]")
    n_valid = sc.notna().sum(axis=1)
    total = sc.mean(axis=1, skipna=True) * 10.0
    return total.where(n_valid >= min_valid_items)


def faq_total_adni_provided(total: pd.Series) -> pd.Series:
    t = pd.to_numeric(total, errors="coerce")
    if ((t < 0) | (t > 30)).any():
        raise ScoringError("FAQ provided total outside 0-30")
    return t


def check_cdr_sb(s: pd.Series, not_assessed_codes=(99,)) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").where(~pd.to_numeric(s, errors="coerce").isin(not_assessed_codes))
    if ((v < 0) | (v > 18)).any() or ((v * 2) % 1 != 0).where(v.notna(), False).any():
        raise ScoringError("CDR sum of boxes outside 0-18 or not in 0.5 steps")
    return v


def recode_sex_female1(s: pd.Series, cohort: str) -> pd.Series:
    """Both ADNI PTGENDER (1=Male, 2=Female) and NACC NACCSEX (1=Male, 2=Female, 8/9 other) -> female=1, male=0; others missing."""
    v = pd.to_numeric(s, errors="coerce")
    if cohort not in ("ADNI", "NACC"):
        raise ScoringError("unknown cohort for sex recode")
    bad = v.notna() & ~v.isin([1, 2, 8, 9])
    if bad.any():
        raise ScoringError("sex codes outside the documented set")
    return v.map({1: 0.0, 2: 1.0})


def age_years_integer(age: pd.Series, lo: float = 18, hi: float = 120) -> pd.Series:
    """Floor to integer years (NACCAGE is an integer age at visit; ADNI age derives from month/year of birth)."""
    a = pd.to_numeric(age, errors="coerce")
    if ((a < lo) | (a > hi)).any():
        raise ScoringError("age outside plausible years range (months or days supplied?)")
    return np.floor(a)


def education_years(s: pd.Series, unknown_codes=(99,), hi: float = 36) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce"); v = v.where(~v.isin(list(unknown_codes)))
    if ((v < 0) | (v > hi)).any():
        raise ScoringError("education outside 0-36 years")
    return v
