"""Entirely synthetic ADNI-shaped cohort for SOFTWARE tests of the multimodal benchmark. No value, identifier, cohort size or
distribution is taken from ADNI, NACC or any participant; nothing produced from it is a research result.

A latent severity drives a conversion hazard, cognition, function, MRI and fluid markers. Deliberate irregularities: missed and
off-schedule visits, annual versus semi-annual schedules by 'phase', reversion to CN, baseline-CN participants, deaths, undated
and same-date-conflicting measurements, values after the data cut-off, MRI QC failures and duplicate same-day scans, assay values
written at the assay bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-08-31")
FS_EXTRA = ["ST101SV", "ST102CV", "ST102SA", "ST102TA", "ST102TS", "ST103CV"]


def make(n: int = 420, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1) + 100000                     # synthetic integer ids (not an ADNI range)
    start = pd.Timestamp("2006-01-01") + pd.to_timedelta(rng.integers(0, 5600, n), unit="D")
    phase = np.where(start.year < 2010, "ADNI1", np.where(start.year < 2014, "ADNI2", "ADNI3"))
    sev = rng.normal(0, 1, n)
    apoe = np.minimum(rng.binomial(2, 1 / (1 + np.exp(-(sev - 0.8)))), 2).astype(float)
    age = rng.normal(73, 7, n).round(1)
    female = rng.binomial(1, 0.45, n).astype(float)
    educ = np.clip(rng.normal(16, 2.8, n).round(), 6, 20)
    conv_years = rng.exponential(1 / (0.22 * np.exp(0.9 * sev + 0.3 * apoe)))
    death = {}
    visits, meas, extra, mri, csf, plasma, amy = [], [], [], [], [], [], []
    for i, pid in enumerate(ids):
        step = 12 if phase[i] == "ADNI3" else 6
        base_cn = rng.random() < 0.04
        dead_at = rng.uniform(0.5, 6) if rng.random() < 0.04 else None
        if dead_at is not None:
            death[int(pid)] = start[i] + pd.Timedelta(days=int(dead_at * 365.25))
        stop = rng.integers(2, 13)                          # dropout after this many scheduled visits
        for j, m in enumerate(range(0, 73, step)):
            if j > stop or (m > 0 and rng.random() < 0.12):
                continue
            t = m / 12
            if dead_at is not None and t > dead_at:
                break
            d = start[i] + pd.Timedelta(days=int(m * 30.44 + (0 if m == 0 else rng.normal(0, 15))))
            if t >= conv_years[i]:
                dx = "MCI" if rng.random() < 0.05 else "DEM"
            else:
                dx = "CN" if (rng.random() < 0.04 or (m == 0 and base_cn)) else "MCI"
            visits.append((pid, m, d, dx, phase[i]))
            L = sev[i] + 0.35 * t + 1.5 * (t >= conv_years[i])
            for var, val in (("mmse", np.clip(round(28 - 1.3 * L + rng.normal(0, 1.2)), 0, 30)),
                             ("adas13", np.clip(14 + 5 * L + rng.normal(0, 3), 0, 85)),
                             ("faq", np.clip(round(2 + 3 * L + rng.normal(0, 2)), 0, 30))):
                if rng.random() < 0.05:
                    continue
                meas.append((pid, var, d + pd.Timedelta(days=int(rng.integers(-6, 7))), val))
            for var, val in (("cdrsb", np.clip(round(2 * (1.5 + L + rng.normal(0, .7))) / 2, 0, 18)),
                             ("cdglobal", 0.5 if L < 2 else 1.0), ("adas11", np.clip(9 + 3.5 * L + rng.normal(0, 2), 0, 70)),
                             ("ravlt_immediate", np.clip(round(38 - 7 * L + rng.normal(0, 6)), 0, 75)),
                             ("ravlt_perc_forgetting", np.clip(45 + 18 * L + rng.normal(0, 15), 0, 100)),
                             ("ldeltotal", np.clip(round(6 - 2.5 * L + rng.normal(0, 2)), 0, 25)),
                             ("trabscor", np.clip(90 + 40 * L + rng.normal(0, 30), 20, 300)),
                             ("catanimsc", np.clip(round(17 - 3 * L + rng.normal(0, 4)), 0, 40))):
                if rng.random() < 0.08:
                    continue
                extra.append((pid, var, d + pd.Timedelta(days=int(rng.integers(-6, 7))), float(val)))
            if m in (0, 12, 24) and rng.random() < (0.78 if phase[i] != "ADNI3" else 0.6):
                sd = d - pd.Timedelta(days=int(rng.integers(3, 45)))
                icv = rng.normal(1.5e6, 1.2e5)
                row = {"RID": pid, "EXAMDATE": sd, "IMAGEUID": int(rng.integers(1, 10**6)),
                       "OVERALLQC": rng.choice(["Pass", "Partial", "Fail"], p=[.85, .1, .05]), "ST10CV": icv}
                for c, mu, b, s in (("ST29SV", 3600, -300, 300), ("ST88SV", 3650, -300, 300), ("ST24CV", 1800, -150, 200),
                                    ("ST83CV", 1750, -150, 200), ("ST24TA", 3.3, -.25, .25), ("ST83TA", 3.3, -.25, .25),
                                    ("ST26CV", 9500, -400, 900), ("ST85CV", 9400, -400, 900), ("ST40CV", 10500, -450, 1000),
                                    ("ST99CV", 10400, -450, 1000), ("ST30SV", 1200, 300, 300), ("ST89SV", 1150, 300, 300),
                                    ("ST37SV", 18000, 5000, 6000), ("ST96SV", 17500, 5000, 6000), ("ST12SV", 1500, -120, 150),
                                    ("ST71SV", 1520, -120, 150)):
                    row[c] = max(rng.normal(mu + b * L, s), 0.1 * mu)
                for c in FS_EXTRA:
                    row[c] = rng.normal(1000 - 50 * L, 100)
                mri.append(row)
                if rng.random() < 0.05:                     # same-day repeat scan
                    mri.append({**row, "IMAGEUID": row["IMAGEUID"] + 1, "OVERALLQC": "Partial"})
            if m == 0 and rng.random() < 0.35:
                a42 = 1200 - 300 * L + rng.normal(0, 150); pt = 20 + 8 * L + rng.normal(0, 4)
                csf.append({"RID": pid, "EXAMDATE": d - pd.Timedelta(days=int(rng.integers(0, 30))),
                            "ABETA42": ">1700" if a42 > 1700 else f"{a42:.1f}", "PTAU": "<8" if pt < 8 else f"{pt:.2f}"})
            if m == 0 and rng.random() < 0.15:
                plasma.append({"RID": pid, "EXAMDATE": d, "pT217_F": max(0.1 + 0.08 * L + rng.normal(0, .04), 0.01),
                               "AB42_AB40_F": 0.1 - 0.005 * L, "NfL_Q": 15 + 4 * L, "GFAP_Q": 150 + 40 * L})
            if m == 0 and rng.random() < 0.4:
                amy.append({"RID": pid, "SCANDATE": d - pd.Timedelta(days=int(rng.integers(0, 60))),
                            "CENTILOIDS": 30 + 25 * L + rng.normal(0, 15)})
    visits = pd.DataFrame(visits, columns=["participant_id", "visit_month", "date", "dx", "phase"])
    measurements = pd.DataFrame(meas, columns=["participant_id", "variable", "date", "value"])
    p0 = int(ids[0])                                         # adversarial rows
    measurements = pd.concat([measurements, pd.DataFrame([
        (p0, "mmse", None, 25), (p0, "mmse", "2030-01-01", 99), (p0, "faq", visits.date.iloc[0], 1),
        (p0, "faq", visits.date.iloc[0], 9)], columns=measurements.columns)], ignore_index=True)
    static = pd.DataFrame({"participant_id": ids, "baseline_age": age, "female": female, "education_years": educ,
                           "apoe4": apoe})
    return {"visits": visits, "measurements": measurements, "static": static, "death_dates": death,
            "extra_long": pd.DataFrame(extra, columns=["participant_id", "variable", "date", "value"]),
            "mri_raw": pd.DataFrame(mri), "csf_raw": pd.DataFrame(csf), "plasma_raw": pd.DataFrame(plasma),
            "amy_raw": pd.DataFrame(amy), "cutoff": CUTOFF}


def datadic(contradict: tuple = ()) -> pd.DataFrame:
    """A synthetic data-dictionary excerpt for UCSFFSX7 (texts written for the test; codes in `contradict` get a wrong region)."""
    text = {"ST10CV": "Volume (Cortical Parcellation) of Icv", "ST29SV": "Volume (WM Parcellation) of LeftHippocampus",
            "ST88SV": "Volume (WM Parcellation) of RightHippocampus", "ST24CV": "Volume (Cortical Parcellation) of LeftEntorhinal",
            "ST83CV": "Volume (Cortical Parcellation) of RightEntorhinal", "ST24TA": "Cortical Thickness Average of LeftEntorhinal",
            "ST83TA": "Cortical Thickness Average of RightEntorhinal", "ST26CV": "Volume (Cortical Parcellation) of LeftFusiform",
            "ST85CV": "Volume (Cortical Parcellation) of RightFusiform",
            "ST40CV": "Volume (Cortical Parcellation) of LeftMiddleTemporal",
            "ST99CV": "Volume (Cortical Parcellation) of RightMiddleTemporal",
            "ST30SV": "Volume (WM Parcellation) of LeftInferiorLateralVentricle",
            "ST89SV": "Volume (WM Parcellation) of RightInferiorLateralVentricle",
            "ST37SV": "Volume (WM Parcellation) of LeftLateralVentricle",
            "ST96SV": "Volume (WM Parcellation) of RightLateralVentricle", "ST12SV": "Volume (WM Parcellation) of LeftAmygdala",
            "ST71SV": "Volume (WM Parcellation) of RightAmygdala"}
    for c in contradict:
        text[c] = "Volume (WM Parcellation) of LeftPutamen"
    return pd.DataFrame({"TBLNAME": "UCSFFSX7", "FLDNAME": list(text), "TEXT": list(text.values())})
