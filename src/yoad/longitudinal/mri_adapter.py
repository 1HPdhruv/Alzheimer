"""MRI adapter for temporal multimodal v1 (PROTOCOL.md §4.2). Pure functions; no file access.

The reused benchmark mapper (multimodal_benchmark.sources.mri_table) decides feature RETENTION from medians computed over
the whole source table, i.e. over participants who later sit in evaluation folds. This adapter separates the two kinds of
rule and does not modify the historical mapper:

  FIXED (prespecified, applied source-wide; no outcome or cohort statistic involved)
    * scans with OVERALLQC 'fail' dropped; several acceptable scans on one date -> best QC grade, then smallest image id
    * a feature whose ST codes the ADNI data dictionary CONTRADICTS is dropped (metadata, not participant values)
    * volume features = summed codes / ICV (ICV <= 0 or missing -> missing); thickness = mean of the codes
  DATA-ADAPTIVE (fitted inside each training fold only, by sequences.Preprocessor)
    * the frozen plausibility rule: a feature whose median RAW value over the training fold's observed MRI assessments lies
      outside multimodal_benchmark.sources.PLAUSIBLE_MEDIAN is inactive in that fold; an implausible training ICV median
      deactivates every volume feature in that fold
The returned table therefore carries the raw summed values (`raw_<feature>`, `raw_icv`) next to the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.multimodal.sources import ICV_CODE, MRI_CORE, PLAUSIBLE_MEDIAN, QC_RANK, SchemaError, _col

RAW_KEYS = ("icv",) + tuple(MRI_CORE)
VOLUME_FEATURES = tuple(f for f, (_c, kind, _k) in MRI_CORE.items() if kind == "volume")


def mri_stream_table(fs: pd.DataFrame, code_status: dict) -> tuple[pd.DataFrame, dict]:
    """participant_id, date, the MRI_CORE features and raw_<key> columns, one row per (participant, date)."""
    rid, date, qc = _col(fs, "RID"), _col(fs, "EXAMDATE", "SCANDATE", "EXAM_DATE"), _col(fs, "OVERALLQC")
    if rid is None or date is None:
        raise SchemaError("FreeSurfer table lacks RID or EXAMDATE")
    img = _col(fs, "IMAGEUID", "IMAGE_UID", "IMAGEID", "LONIUID")
    t = fs.copy()
    q = t[qc].astype(str).str.strip().str.lower() if qc else pd.Series("not recorded", index=t.index)
    info = {"rows_in": int(len(t)), "rows_qc_fail": int((q == "fail").sum())}
    t = t[q != "fail"].copy(); q = q[q != "fail"]
    t["_qc"] = q.map(QC_RANK).fillna(2).to_numpy()
    t["_img"] = pd.to_numeric(t[img], errors="coerce") if img else np.nan
    num = lambda c: pd.to_numeric(t[_col(t, c)], errors="coerce") if _col(t, c) else pd.Series(np.nan, index=t.index)
    icv = num(ICV_CODE)
    icv = icv.where(np.isfinite(icv) & (icv > 0))
    feats, raw, status = {}, {"raw_icv": icv}, {}
    for f, (codes, kind, _kw) in MRI_CORE.items():
        bad = [c for c in codes + ((ICV_CODE,) if kind == "volume" else ()) if str(code_status.get(c, "")).startswith("contradicted")]
        if bad or any(_col(t, c) is None for c in codes):
            status[f] = "dropped (fixed rule): contradicted or absent ST code"
            continue
        parts = pd.concat([num(c) for c in codes], axis=1)
        s = parts.sum(axis=1, min_count=len(codes)) if kind == "volume" else parts.mean(axis=1, skipna=False)
        s = s.where(np.isfinite(s))
        raw[f"raw_{f}"] = s
        feats[f] = s / icv if kind == "volume" else s
        status[f] = "kept by fixed rules; plausibility decided inside each training fold"
    out = pd.DataFrame({"participant_id": pd.to_numeric(t[rid], errors="coerce"), "date": pd.to_datetime(t[date], errors="coerce"),
                        "_qc": t["_qc"], "_img": t["_img"], **feats, **raw})
    out = out.dropna(subset=["participant_id", "date"])
    n0 = len(out)
    out = (out.sort_values(["participant_id", "date", "_qc", "_img"], kind="mergesort", na_position="last")
              .drop_duplicates(["participant_id", "date"], keep="first"))
    info.update(rows_used=int(len(out)), same_date_scans_resolved=int(n0 - len(out)), feature_status=status)
    for k in RAW_KEYS:                                       # every raw column present, even for a dropped feature
        if f"raw_{k}" not in out.columns:
            out[f"raw_{k}"] = np.nan
    return out.drop(columns=["_qc", "_img"]).reset_index(drop=True), info


def plausible_in_fold(raw_medians: dict) -> dict:
    """{feature: active} from TRAINING-fold medians of the raw values (the frozen PLAUSIBLE_MEDIAN ranges)."""
    icv = raw_medians.get("icv")
    icv_ok = icv is not None and np.isfinite(icv) and PLAUSIBLE_MEDIAN["icv"][0] <= icv <= PLAUSIBLE_MEDIAN["icv"][1]
    out = {}
    for f in MRI_CORE:
        m = raw_medians.get(f)
        ok = m is not None and np.isfinite(m) and PLAUSIBLE_MEDIAN[f][0] <= m <= PLAUSIBLE_MEDIAN[f][1]
        out[f] = bool(ok and (icv_ok or f not in VOLUME_FEATURES))
    return out
