"""ADNI table mappers for the instruments the benchmark adds to the frozen predictors (PROTOCOL.md §4).

Pure functions on in-memory tables, except `read_table`, which only the guarded runner calls. Column names come from the
header-only inventory (reports/schema_headers.csv, 2026-10-12) and the ADNIMERGE2 documentation; every mapper checks them and
returns a recorded reason instead of guessing.

Long-format instruments (attached with the FROZEN src/cohort/anchor_predictors.attach_predictors rule: nearest assessment in
the window, ties to the earlier date, undated / post-cut-off / same-date-conflicting values removed):
  CDR (CDR_12Sep2026.csv): cdrsb = CDRSB, cdglobal = CDGLOBAL;  ADAS.csv: adas11 = TOTSCORE;
  NEUROBAT.csv: ravlt_immediate = AVTOT1 + ... + AVTOT5 (all five valid), ravlt_perc_forgetting = 100 (AVTOT5 - AVDEL30MIN) /
  AVTOT5 (AVTOT5 > 0), ldeltotal = LDELTOTAL, trabscor = TRABSCOR, catanimsc = CATANIMSC. Negative ADNI codes -> missing.
Wide instruments (one assessment = one participant and date; attached by cohort.attach_instrument):
  FreeSurfer 7 (UCSFFSX7): ICV-normalised regional volumes and entorhinal thickness (MRI_CORE) plus the Simkhada Table S1
  columns (fs_<code>). Scans with OVERALLQC 'Fail' are dropped; several acceptable scans on one date -> the best QC grade, then
  the smallest image id (independent of row order). The ST-code mapping is checked against the ADNI data dictionary and against
  prespecified plausibility ranges of the median raw value (mm^3 / mm); a contradicted or implausible feature is dropped.
  CSF Roche Elecsys: ABETA42, PTAU (values written '<x' / '>x' are set to x and counted); plasma Fujirebio/Quanterix: pT217_F,
  AB42_AB40_F, NfL_Q, GFAP_Q; amyloid PET (UCBERKELEY_AMY_6MM): CENTILOIDS.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

LONG_MAP = {"cdrsb": ("cdr", "CDRSB"), "cdglobal": ("cdr", "CDGLOBAL"), "adas11": ("adas", "TOTSCORE"),
            "ldeltotal": ("neurobat", "LDELTOTAL"), "trabscor": ("neurobat", "TRABSCOR"), "catanimsc": ("neurobat", "CATANIMSC")}
RAVLT_TRIALS = ("AVTOT1", "AVTOT2", "AVTOT3", "AVTOT4", "AVTOT5")
MRI_CORE = {   # feature -> (ST codes, kind, required dictionary keyword per code); volumes are summed and divided by ICV
    "mri_hippocampus": (("ST29SV", "ST88SV"), "volume", ("lefthippocampus", "righthippocampus")),
    "mri_entorhinal": (("ST24CV", "ST83CV"), "volume", ("leftentorhinal", "rightentorhinal")),
    "mri_entorhinal_thickness": (("ST24TA", "ST83TA"), "thickness", ("leftentorhinal", "rightentorhinal")),
    "mri_fusiform": (("ST26CV", "ST85CV"), "volume", ("leftfusiform", "rightfusiform")),
    "mri_midtemp": (("ST40CV", "ST99CV"), "volume", ("leftmiddletemporal", "rightmiddletemporal")),
    "mri_ventricles": (("ST30SV", "ST37SV", "ST89SV", "ST96SV"), "volume",
                       ("leftinferiorlateralventricle", "leftlateralventricle", "rightinferiorlateralventricle",
                        "rightlateralventricle")),
    "mri_amygdala": (("ST12SV", "ST71SV"), "volume", ("leftamygdala", "rightamygdala")),
}
ICV_CODE, ICV_KEYWORD = "ST10CV", "icv"
PLAUSIBLE_MEDIAN = {   # raw (un-normalised) median of the summed codes; prespecified ranges for adults aged 55-90
    "icv": (0.9e6, 2.2e6), "mri_hippocampus": (3000, 12000), "mri_entorhinal": (1500, 7000),
    "mri_entorhinal_thickness": (1.5, 5.0), "mri_fusiform": (10000, 32000), "mri_midtemp": (10000, 36000),
    "mri_ventricles": (5000, 200000), "mri_amygdala": (1500, 5500)}
QC_RANK = {"pass": 0, "partial": 1}     # anything else that is not 'fail' ranks 2; 'fail' is dropped


class SchemaError(ValueError):
    """A table lacks the columns a mapper needs; the modality is recorded as unavailable for that reason."""


def read_table(path: str) -> tuple[pd.DataFrame, dict]:
    """CSV, or an R data file (.rda / .RData / .rds) through pyreadr. Returns (table, file identity). Runner only."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ident = {"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": h.hexdigest()}
    if path.lower().endswith((".rda", ".rdata", ".rds")):
        import pyreadr
        res = pyreadr.read_r(path)
        if len(res) != 1:
            raise SchemaError(f"{ident['file']}: expected one object, found {len(res)}")
        t = next(iter(res.values()))
    else:
        t = pd.read_csv(path, low_memory=False)
    for c in t.columns:                                     # R factors -> plain strings
        if isinstance(t[c].dtype, pd.CategoricalDtype):
            t[c] = t[c].astype(object).where(t[c].notna(), None)
    return t, ident


def _col(t: pd.DataFrame, *names):
    for n in names:
        for c in t.columns:
            if c.upper() == n.upper():
                return c
    return None


def _num(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where(v >= 0)


def _rid(t: pd.DataFrame, name: str) -> pd.Series:
    c = _col(t, "RID", "SUBJID")
    if c is None:
        raise SchemaError(f"{name}: no RID column")
    return pd.to_numeric(t[c], errors="coerce")


def long_measurements(tables: dict) -> tuple[pd.DataFrame, dict]:
    """Long table (participant_id, variable, date, value) of the added long-format instruments, from whichever of
    tables['cdr'], ['adas'], ['neurobat'] are supplied. Returns (measurements, {variable: reason}) for unavailable ones."""
    out, missing = [], {}
    for var, (tab, col) in LONG_MAP.items():
        t = tables.get(tab)
        if t is None:
            missing[var] = f"{tab} table not supplied"; continue
        c, d = _col(t, col), _col(t, "VISDATE", "EXAMDATE")
        if c is None or d is None:
            missing[var] = f"{tab} table lacks {col} or VISDATE"; continue
        out.append(pd.DataFrame({"participant_id": _rid(t, tab), "variable": var, "date": t[d], "value": _num(t[c])}))
    nb = tables.get("neurobat")
    if nb is not None and all(_col(nb, c) for c in RAVLT_TRIALS + ("AVDEL30MIN", "VISDATE")):
        trials = pd.concat([_num(nb[_col(nb, c)]) for c in RAVLT_TRIALS], axis=1)
        imm = trials.sum(axis=1, min_count=5)                # all five trials valid, else missing
        t5, d30 = trials.iloc[:, 4], _num(nb[_col(nb, "AVDEL30MIN")])
        pf = (100.0 * (t5 - d30) / t5).where(t5 > 0)
        pid, date = _rid(nb, "neurobat"), nb[_col(nb, "VISDATE")]
        out += [pd.DataFrame({"participant_id": pid, "variable": "ravlt_immediate", "date": date, "value": imm}),
                pd.DataFrame({"participant_id": pid, "variable": "ravlt_perc_forgetting", "date": date, "value": pf})]
    else:
        missing.update({"ravlt_immediate": "NEUROBAT RAVLT columns unavailable",
                        "ravlt_perc_forgetting": "NEUROBAT RAVLT columns unavailable"})
    if not out:
        return pd.DataFrame(columns=["participant_id", "variable", "date", "value"]), missing
    m = pd.concat(out, ignore_index=True).dropna(subset=["participant_id"])
    return m, missing


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def verify_mri_codes(datadic: pd.DataFrame | None, table: str = "UCSFFSX7") -> dict:
    """Checks every MRI_CORE code (and ICV) against the data dictionary rows of `table`. Returns {code: 'verified' |
    'contradicted: <text>' | 'not in dictionary'}; no participant data are involved."""
    codes = {ICV_CODE: (ICV_KEYWORD, "volume")}
    for feat, (cs, kind, kws) in MRI_CORE.items():
        codes.update({c: (k, kind) for c, k in zip(cs, kws)})
    if datadic is None:
        return {c: "not in dictionary" for c in codes}
    tb, fl, tx = _col(datadic, "TBLNAME"), _col(datadic, "FLDNAME"), _col(datadic, "TEXT")
    if not (tb and fl and tx):
        return {c: "not in dictionary" for c in codes}
    d = datadic[datadic[tb].astype(str).str.upper() == table.upper()]
    text = {str(f).upper(): t for f, t in zip(d[fl], d[tx])}
    out = {}
    for c, (kw, kind) in codes.items():
        if c not in text:
            out[c] = "not in dictionary"
        else:
            t = _norm(text[c])
            out[c] = "verified" if (kw in t and kind in t) else f"contradicted: {str(text[c])[:80]}"
    return out


def mri_table(fs: pd.DataFrame, table_s1: list, code_status: dict) -> tuple[pd.DataFrame, dict]:
    """Wide MRI assessments: participant_id, date, MRI_CORE features (ICV-normalised volumes; thickness in mm) and fs_<code>
    Table S1 columns. Returns (table, info) with QC counts, the feature status and the plausibility medians (aggregate)."""
    rid, date, qc = _col(fs, "RID"), _col(fs, "EXAMDATE", "SCANDATE", "EXAM_DATE"), _col(fs, "OVERALLQC")
    if rid is None or date is None:
        raise SchemaError("FreeSurfer table lacks RID or EXAMDATE")
    img = _col(fs, "IMAGEUID", "IMAGE_UID", "IMAGEID", "LONIUID")
    t = fs.copy()
    q = t[qc].astype(str).str.strip().str.lower() if qc else pd.Series("not recorded", index=t.index)
    info = {"rows_in": int(len(t)), "qc_column": qc or "absent", "rows_qc_fail": int((q == "fail").sum())}
    t = t[q != "fail"].copy(); q = q[q != "fail"]
    t["_qc"] = q.map(QC_RANK).fillna(2).to_numpy()
    t["_img"] = pd.to_numeric(t[img], errors="coerce") if img else np.nan
    num = lambda c: pd.to_numeric(t[_col(t, c)], errors="coerce") if _col(t, c) else pd.Series(np.nan, index=t.index)
    raw = {}
    icv = num(ICV_CODE)
    raw["icv"] = icv
    feats, status = {}, {}
    for feat, (cs, kind, _) in MRI_CORE.items():
        bad = [c for c in cs + ((ICV_CODE,) if kind == "volume" else ()) if str(code_status.get(c, "")).startswith("contradicted")]
        missing_cols = [c for c in cs if _col(t, c) is None]
        if bad or missing_cols:
            status[feat] = f"dropped: {'contradicted ' + ','.join(bad) if bad else 'columns absent ' + ','.join(missing_cols)}"
            continue
        parts = pd.concat([num(c) for c in cs], axis=1)
        s = parts.sum(axis=1, min_count=len(cs)) if kind == "volume" else parts.mean(axis=1, skipna=False)
        raw[feat] = s
        feats[feat] = s / icv.where(icv > 0) if kind == "volume" else s
        st = {code_status.get(c, "not in dictionary") for c in cs}
        status[feat] = "verified" if st == {"verified"} else "unverified (codes not in dictionary)"
    medians = {k: (float(v.median()) if v.notna().any() else None) for k, v in raw.items()}
    lo_hi = PLAUSIBLE_MEDIAN
    if medians["icv"] is None or not (lo_hi["icv"][0] <= medians["icv"] <= lo_hi["icv"][1]):
        for f in list(feats):                                  # ICV implausible -> no normalised volume can be trusted
            if MRI_CORE[f][1] == "volume":
                feats.pop(f); status[f] = "dropped: ICV absent or its median implausible"
    for f in list(feats):
        m = medians.get(f)
        if m is None or not (lo_hi[f][0] <= m <= lo_hi[f][1]):
            feats.pop(f); status[f] = "dropped: median outside the prespecified plausible range"
    full = [c for c in table_s1 if _col(t, c)]
    out = pd.DataFrame({"participant_id": pd.to_numeric(t[rid], errors="coerce"), "date": pd.to_datetime(t[date], errors="coerce"),
                        "_qc": t["_qc"], "_img": t["_img"], **feats, **{f"fs_{c}": num(c) for c in full}})
    out = out.dropna(subset=["participant_id", "date"])
    n_before = len(out)
    out = (out.sort_values(["participant_id", "date", "_qc", "_img"], kind="mergesort", na_position="last")
              .drop_duplicates(["participant_id", "date"], keep="first"))
    info.update(rows_used=int(len(out)), same_date_scans_resolved=int(n_before - len(out)), feature_status=status,
                plausibility_medians={k: (None if v is None else round(v, 4)) for k, v in medians.items()},
                table_s1_columns_present=len(full), table_s1_columns_expected=len(table_s1))
    return out.drop(columns=["_qc", "_img"]).reset_index(drop=True), info


def _bounded(s: pd.Series) -> tuple[pd.Series, int]:
    txt = s.astype(str).str.strip()
    flagged = txt.str.match(r"^[<>]")
    v = pd.to_numeric(txt.str.lstrip("<>").str.strip(), errors="coerce")
    return v.where(v >= 0), int(flagged.sum())


def csf_table(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = _col(t, "EXAMDATE", "DRAWDATE", "DRAWDTE")
    a42, pt = _col(t, "ABETA42"), _col(t, "PTAU")
    if not (d and a42 and pt):
        raise SchemaError("CSF table lacks EXAMDATE, ABETA42 or PTAU")
    va, na = _bounded(t[a42]); vp, npt = _bounded(t[pt])
    out = pd.DataFrame({"participant_id": _rid(t, "csf"), "date": pd.to_datetime(t[d], errors="coerce"),
                        "csf_abeta42": va, "csf_ptau": vp, "csf_log_ptau_abeta42": np.log(vp / va.where(va > 0))})
    return out, {"values_at_assay_bound_abeta42": na, "values_at_assay_bound_ptau": npt}


PLASMA_MAP = {"plasma_ptau217": "pT217_F", "plasma_ab42_ab40": "AB42_AB40_F", "plasma_nfl": "NfL_Q", "plasma_gfap": "GFAP_Q"}


def plasma_table(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = _col(t, "EXAMDATE")
    if not d or not all(_col(t, c) for c in PLASMA_MAP.values()):
        raise SchemaError("plasma table lacks EXAMDATE or an assay column")
    out = pd.DataFrame({"participant_id": _rid(t, "plasma"), "date": pd.to_datetime(t[d], errors="coerce"),
                        **{k: _bounded(t[_col(t, c)])[0] for k, c in PLASMA_MAP.items()}})
    return out, {}


def amyloid_table(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d, c = _col(t, "SCANDATE", "EXAMDATE"), _col(t, "CENTILOIDS", "CENTILOID")
    if not (d and c):
        raise SchemaError("amyloid PET table lacks SCANDATE or CENTILOIDS")
    v = pd.to_numeric(t[c], errors="coerce")                     # centiloids may be negative: no sign filter
    out = pd.DataFrame({"participant_id": _rid(t, "amyloid"), "date": pd.to_datetime(t[d], errors="coerce"), "amy_centiloid": v})
    return out, {}
