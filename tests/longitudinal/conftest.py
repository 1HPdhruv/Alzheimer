"""Shared synthetic fixtures. Every frame comes from multimodal_benchmark/tests/synthetic_cohort.py (no ADNI or NACC value,
identifier, size or distribution) passed through the FROZEN cohort functions and this module's stream builders. Nothing here is
a research result."""
from __future__ import annotations

import numpy as np
import pytest

from yoad.multimodal import cohort as C, sources as SRC
from tests.multimodal import synthetic_cohort as syn
from yoad.longitudinal import mri_adapter as MA
from yoad.longitudinal import sequences as SQ


def wide_tables(d: dict) -> tuple[dict, dict]:
    """(frozen-mapper wide tables for build_frames, stream tables with the fold-safe MRI adapter)."""
    codes = SRC.verify_mri_codes(syn.datadic())
    mri, _ = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, codes)
    ms, _ = MA.mri_stream_table(d["mri_raw"], codes)
    csf, _ = SRC.csf_table(d["csf_raw"]); pl, _ = SRC.plasma_table(d["plasma_raw"]); am, _ = SRC.amyloid_table(d["amy_raw"])
    f = lambda t: [c for c in t.columns if c not in ("participant_id", "date")]
    wide = {"mri": (mri, f(mri)), "csf": (csf, f(csf)), "plasma": (pl, f(pl)), "amy": (am, f(am))}
    return wide, dict(wide, mri=(ms, f(ms)))


def build(d: dict, flagged=frozenset()) -> dict:
    wide, swide = wide_tables(d)
    fr = C.build_frames(d["visits"], d["measurements"], d["static"], d["cutoff"], d["death_dates"], d["extra_long"], wide)
    rows = fr["rows"]["strict"]
    streams, sflow = SQ.dated_streams(d["measurements"], d["extra_long"], swide, d["cutoff"])
    s = SQ.build_samples(rows, streams, "state_REF", flagged)
    part = s.meta["partition"].to_numpy()
    dev_rows = rows[rows["partition"] == "train"]
    return {"d": d, "rows": rows, "fr": fr, "streams": streams, "sflow": sflow, "swide": swide, "s": s,
            "dev": s.subset(np.flatnonzero(part == "train")), "heldout": s.subset(np.flatnonzero(part == "held_out")),
            "folds": C.outer_folds(dev_rows[dev_rows["is_baseline"]])}


@pytest.fixture(scope="session")
def cohort():
    return build(syn.make(320, seed=11))
