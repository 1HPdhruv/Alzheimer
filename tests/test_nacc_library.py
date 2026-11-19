import numpy as np
import pandas as pd
import pytest
from yoad.data import nacc_mapper as nm
from yoad.experiments.nacc_validation import validation_lib as vl

def nacc_rows():
    base = {c: 0 for c in nm.FAQ_ITEMS}
    r = []
    def add(pid, vnum, date, udsd, mmse=28, **kw):
        r.append({"NACCID": pid, "NACCVNUM": vnum, "VISITDATE": date, "NACCUDSD": udsd, "NACCAGE": 75, "NACCSEX": 2, "EDUC": 16,
                  "HANDED": 2, "NACCMMSE": mmse, "CDRGLOB": 0.5, "CDRSUM": 1.5, "NACCNE4S": 1, **base, **kw})
    add("A", 1, "2010-01-01", 3); add("A", 2, "2011-01-05", 3); add("A", 3, "2011-12-20", 4)            # converter (dementia before H)
    add("B", 1, "2010-01-01", 3, HANDED=1, NACCSEX=1, BILLS=8); add("B", 2, "2011-01-01", 3); add("B", 3, "2012-01-01", 2)  # stable
    add("C", 1, "2015-01-01", 3, mmse=-4)                                                              # MoCA era, no follow-up
    add("D", 1, "2010-01-01", 1); add("D", 2, "2011-01-01", 3)                                         # normal at entry
    add("E", 1, "2010-01-01", 3, EDUC=99, HANDED=3, NACCNE4S=9, BILLS=9, TAXES=9, SHOPPING=9)          # missing codes
    return pd.DataFrame(r)

def test_nacc_mapping_codes_and_visit_months():
    v, f, flow = nm.map_nacc(nacc_rows())
    assert dict(zip(v.participant_id + v.visit_month.astype(str), v.dx))["A24"] == "DEM"
    assert set(v[v.participant_id == "B"].dx) == {"MCI", "CN"}                  # impaired-not-MCI -> non-dementia
    fb = f[(f.participant_id == "B") & (f.visit_month == 0)].iloc[0]
    assert fb.PTGENDER == 0 and fb.PTHAND == 1 and fb.FAQTOTAL == 0               # one item 'not applicable' -> prorated
    fe = f[f.participant_id == "E"].iloc[0]
    assert np.isnan(fe.PTEDUCAT) and np.isnan(fe.PTHAND) and np.isnan(fe.APOE4) and np.isnan(fe.FAQTOTAL)   # 3 items missing
    assert np.isnan(f[f.participant_id == "C"].iloc[0].MMSCORE) and flow["handedness_ambidextrous_or_unknown"] == 1
    with pytest.raises(ValueError, match="CDRGLOB"):
        nm.map_nacc(nacc_rows().drop(columns=["CDRGLOB"]))

def test_nacc_calendar_labels_use_the_same_rule_and_window():
    v, f, _ = nm.map_nacc(nacc_rows())
    s, flow = nm.nacc_samples(v, f, "2020-01-01", window_days=180)
    lab = {(p, int(m)): int(y) for p, m, y in zip(s.participant_id, s.anchor_month, s.y)}
    assert lab[("A", 0)] == 1 and lab[("B", 0)] == 0 and ("C", 0) not in lab        # C has no follow-up -> indeterminate
    assert ("A", 12) in lab and set(s.columns) >= set(nm.FEATURES)
    late = nacc_rows(); late.loc[late.VISITDATE == "2011-12-20", "VISITDATE"] = "2012-01-10"      # dementia 9 days after H
    v2, f2, _ = nm.map_nacc(late)
    assert ("A", 0) not in {(p, int(m)) for p, m in zip(*nm.nacc_samples(v2, f2, "2020-01-01", 180)[0][["participant_id", "anchor_month"]].T.values)}
    s3, _ = nm.nacc_samples(v2, f2, "2020-01-01", 180, event_grace_days=28)
    assert dict(zip(zip(s3.participant_id, s3.anchor_month), s3.y))[("A", 0)] == 1      # counted with an event grace period

def synth(n, seed, shift=0.0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        z = rng.normal() + shift
        for v in range(int(rng.integers(1, 3))):
            rows.append({"participant_id": f"{seed}-{i}", "anchor_month": 12 * v, "y": int(rng.random() < 1 / (1 + np.exp(-(1.5 * z - .5)))),
                         "MMSCORE": 27 - z + rng.normal(), "CDGLOBAL": 0.5, "CDRSB": 1.5 + z + rng.normal(0, .5),
                         "FAQTOTAL": max(0, 3 + 2 * z + rng.normal()), "AGE": 73 + rng.normal(0, 6), "PTEDUCAT": 16.0,
                         "APOE4": float(rng.integers(0, 3)), "PTGENDER": float(rng.integers(0, 2)), "PTHAND": float(rng.random() < .1)})
    return pd.DataFrame(rows)

def test_develop_on_adni_and_validate_frozen_model_on_nacc():
    num, cat = vl.columns("NACC_T")
    adni, nacc = synth(120, 1), synth(80, 2, shift=0.3)
    nacc.index = pd.RangeIndex(10**9, 10**9 + len(nacc))
    cal, thr, info = vl.develop(adni, num, cat, "logistic", "none", n_iter=3, outer=(3, 1))
    rows = info.pop("fitted_rows")
    assert rows <= set(adni.index) and 0 < thr < 1
    res = vl.validate(cal, thr, nacc, num, cat, B=50, fitted_rows=rows)
    assert {"roc_auc", "log_oe", "calibration_slope", "balanced_accuracy"} <= set(res["visit_level"])
    assert "log_oe" in set(res["visit_level_ci"].metric) and res["n_participants"] == 80
    assert res["first_visit"] is None or res["n_first_visit_samples"] == 80
    # validation never refits: the same frozen model gives the same results, and NACC labels do not affect the threshold
    flipped = nacc.assign(y=1 - nacc.y)
    assert vl.validate(cal, thr, flipped, num, cat, B=10, fitted_rows=rows)["visit_level"]["roc_auc"] == pytest.approx(1 - res["visit_level"]["roc_auc"])
    with pytest.raises(AssertionError):
        vl.validate(cal, thr, adni.iloc[:30], num, cat, B=5, fitted_rows=rows)            # rows used for fitting refused
    cm = vl.case_mix(adni, nacc, num + cat)
    assert list(cm.cohort) == ["ADNI", "NACC"] and "MMSCORE_mean" in cm
