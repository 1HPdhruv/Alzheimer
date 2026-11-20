"""Synthetic positive control for the temporal architectures. The outcome depends only on the TRAJECTORY of one cognitive
feature (its slope over the history); the anchor-visit value carries no information. A temporal model must learn it and the
anchor-only input must not. This checks that a null history effect on real data could not simply be an architecture that is
unable to read sequences. It is a software property, not a research result."""
from __future__ import annotations

import numpy as np
import pytest

from yoad.multimodal.metrics import auroc
from yoad.longitudinal import config as CF
from yoad.longitudinal import models as MD
from yoad.longitudinal import train as TR


def make(n=1400, L=4, seed=0):
    rng = np.random.default_rng(seed)
    a = {"lengths": np.full(n, L), "valid": np.ones((n, L), bool),
         "dt": np.tile(np.array([-1.5, -1.0, -0.5, 0.0], np.float32), (n, 1)), "gap": np.tile(np.array([0, .5, .5, .5],
                                                                                                       np.float32), (n, 1))}
    slope = rng.normal(0, 1, n)
    current = rng.normal(0, 1, n)
    y = (slope + rng.normal(0, 0.3, n) > 0).astype(float)
    for b, feats in CF.TV_BLOCKS.items():
        d = len(feats)
        a[f"x_{b}"] = np.zeros((n, L, d), np.float32); a[f"m_{b}"] = np.zeros((n, L, d), np.float32)
        a[f"lx_{b}"] = np.zeros((n, d), np.float32); a[f"lm_{b}"] = np.zeros((n, d), np.float32)
        a[f"ls_{b}"] = np.zeros((n, d), np.float32)
    a["x_COGFUNC"][:, :, 0] = (current[:, None] + slope[:, None] * a["dt"]).astype(np.float32)   # x_t = current + slope*dt
    a["m_COGFUNC"][:, :, 0] = 1.0
    a["lx_COGFUNC"][:, 0] = current; a["lm_COGFUNC"][:, 0] = 1.0
    a["x_STATIC"] = rng.normal(0, 1, (n, 4)).astype(np.float32); a["m_STATIC"] = np.ones((n, 4), np.float32)
    for b, feats in CF.TV_BLOCKS.items():                       # per-modality streams (Model E): cognition only
        a[f"sx_{b}"] = a[f"x_{b}"].copy(); a[f"sm_{b}"] = a[f"m_{b}"].copy()
        a[f"st_{b}"] = a["dt"].copy(); a[f"sg_{b}"] = a["gap"].copy(); a[f"sv_{b}"] = a[f"m_{b}"].sum(-1) > 0
    return a, y


def anchor_only(a):
    b = {k: v.copy() for k, v in a.items()}
    b["x_COGFUNC"] = np.concatenate([a["x_COGFUNC"][:, -1:], np.zeros_like(a["x_COGFUNC"][:, 1:])], 1)
    b["m_COGFUNC"] = np.concatenate([a["m_COGFUNC"][:, -1:], np.zeros_like(a["m_COGFUNC"][:, 1:])], 1)
    b["valid"] = np.zeros_like(a["valid"]); b["valid"][:, 0] = True
    b["lengths"] = np.ones_like(a["lengths"]); b["dt"] = np.zeros_like(a["dt"]); b["gap"] = np.zeros_like(a["gap"])
    for key in ("sx_COGFUNC", "sm_COGFUNC"):
        b[key] = np.concatenate([a[key][:, -1:], np.zeros_like(a[key][:, 1:])], 1)
    b["st_COGFUNC"] = np.zeros_like(a["st_COGFUNC"]); b["sg_COGFUNC"] = np.zeros_like(a["sg_COGFUNC"])
    b["sv_COGFUNC"] = np.zeros_like(a["sv_COGFUNC"]); b["sv_COGFUNC"][:, 0] = True
    return b


@pytest.mark.parametrize("mid", ("B_LSTM", "C_TRANSFORMER", "E_COMBINED"))
def test_temporal_models_learn_a_trajectory_signal_that_the_anchor_visit_lacks(mid, monkeypatch):
    monkeypatch.setattr(CF, "NEURAL_FIXED", dict(CF.NEURAL_FIXED, max_epochs=60, patience=10))
    a, y = make()
    w = np.ones(len(y))
    tr, va, te = np.arange(0, 900), np.arange(900, 1100), np.arange(1100, 1400)
    full = TR.fit(mid, 16, a, y, w, tr, va, seed=0)["model"]
    snap_a = anchor_only(a)
    snap = TR.fit(mid, 16, snap_a, y, w, tr, va, seed=0)["model"]
    au_full = auroc(y[te], TR.predict(full, a, te))
    au_snap = auroc(y[te], TR.predict(snap, snap_a, te))
    assert au_full > 0.85, f"{mid} failed to learn the trajectory ({au_full:.3f})"
    assert au_snap < 0.65, f"anchor-only input should be uninformative ({au_snap:.3f})"
