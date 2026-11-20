"""Synthetic tests: neural architectures — shapes, invariance to absent modalities and padding, modality dropout, gating,
parameter budget, determinism and training behaviour."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from yoad.longitudinal import config as CF
from yoad.longitudinal import models as MD
from yoad.longitudinal import sequences as SQ
from yoad.longitudinal import train as TR

IDS = ("B_LSTM", "C_TRANSFORMER", "D_FUSION", "E_COMBINED")


@pytest.fixture(scope="module")
def arrays(cohort):
    s = cohort["dev"]
    return s, SQ.Preprocessor().fit(s, np.arange(s.n)).transform(s)


def _eval(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch).numpy()


@pytest.mark.parametrize("mid", IDS)
def test_forward_shapes_and_finite(arrays, mid):
    s, a = arrays
    torch.manual_seed(0)
    m = MD.build(mid, 16)
    out = _eval(m, TR.tensors(a, np.arange(40)))
    assert out.shape == (40,) and np.isfinite(out).all()


@pytest.mark.parametrize("mid", IDS)
def test_absent_modality_values_do_not_matter(arrays, mid):
    s, a = arrays
    torch.manual_seed(0)
    m = MD.build(mid, 16)
    idx = np.arange(60)
    b1 = TR.tensors(a, idx)
    b2 = {k: v.clone() for k, v in b1.items()}
    for blk in CF.TV_BLOCKS:                                # write garbage wherever a modality is absent at a token
        absent = (b1[f"m_{blk}"].sum(-1) == 0).unsqueeze(-1)
        b2[f"x_{blk}"] = torch.where(absent, torch.full_like(b1[f"x_{blk}"], 7.5), b1[f"x_{blk}"])
        s_abs = (~b1[f"sv_{blk}"].bool()).unsqueeze(-1)       # stream positions without an observed assessment
        b2[f"sx_{blk}"] = torch.where(s_abs, torch.full_like(b1[f"sx_{blk}"], -6.0), b1[f"sx_{blk}"])
        b2[f"st_{blk}"] = torch.where(s_abs[..., 0], torch.full_like(b1[f"st_{blk}"], -9.0), b1[f"st_{blk}"])
        lab = (b1[f"lm_{blk}"].sum(-1, keepdim=True) == 0)
        b2[f"lx_{blk}"] = torch.where(lab, torch.full_like(b1[f"lx_{blk}"], -3.0), b1[f"lx_{blk}"])
        b2[f"ls_{blk}"] = torch.where(lab, torch.full_like(b1[f"ls_{blk}"], 9.0), b1[f"ls_{blk}"])
    assert np.allclose(_eval(m, b1), _eval(m, b2), atol=1e-5)


@pytest.mark.parametrize("mid", ("B_LSTM", "C_TRANSFORMER"))
def test_padding_positions_do_not_matter(arrays, mid):
    s, a = arrays
    torch.manual_seed(1)
    m = MD.build(mid, 16)
    idx = np.flatnonzero(a["lengths"] < CF.MAX_DATES)[:60]
    b1 = TR.tensors(a, idx)
    b2 = {k: v.clone() for k, v in b1.items()}
    pad = ~b1["valid"]
    for blk in CF.TV_BLOCKS:
        b2[f"x_{blk}"][pad] = 5.0
        b2[f"m_{blk}"][pad] = 1.0                           # even a (wrong) mask at padding must be ignored
    b2["dt"][pad] = -3.0; b2["gap"][pad] = 2.0
    assert np.allclose(_eval(m, b1), _eval(m, b2), atol=1e-5)


def test_modality_dropout_training_only_and_never_all(arrays):
    s, a = arrays
    b = TR.tensors(a, np.arange(s.n))
    assert MD.modality_dropout(b, 0.9, torch.Generator().manual_seed(0), training=False) is b
    out = MD.modality_dropout(b, 0.95, torch.Generator().manual_seed(0), training=True)
    have = torch.stack([MD.present(b, x).amax(1) for x in MD.TV], 1)
    left = torch.stack([MD.present(out, x).amax(1) for x in MD.TV], 1)
    assert ((have.sum(1) == 0) | (left.sum(1) > 0)).all()      # an anchor with data keeps at least one modality
    assert (left <= have).all()                               # dropout only removes, never invents
    assert left.sum() < have.sum()


def test_gate_weights_zero_for_absent_and_sum_to_one(arrays):
    s, a = arrays
    torch.manual_seed(0)
    m = MD.build("D_FUSION", 16)
    b = TR.tensors(a, np.arange(80))
    _eval(m, b)
    w = m.last_gate_.numpy()
    assert np.allclose(w.sum(1), 1.0, atol=1e-5)
    for i, blk in enumerate(MD.TV):
        absent = (b[f"lm_{blk}"].sum(-1) == 0).numpy()
        assert np.allclose(w[absent, i], 0.0)


def test_parameter_budget():
    for mid in IDS:
        for h in (16, 32):
            assert MD.n_params(MD.build(mid, h)) < 25_000


def test_training_is_deterministic_and_learns(arrays):
    s, a = arrays
    y = s.meta["y"].to_numpy(float); w = SQ.weights(s.meta["participant_id"].to_numpy())
    tr, va = np.arange(0, s.n, 2), np.arange(1, s.n, 2)
    f1 = TR.fit("D_FUSION", 16, a, y, w, tr, va, seed=5)
    f2 = TR.fit("D_FUSION", 16, a, y, w, tr, va, seed=5)
    assert np.allclose(f1["p_val"], f2["p_val"]) and f1["best_epoch"] == f2["best_epoch"]
    from yoad.multimodal.metrics import auroc
    assert auroc(y[va], f1["p_val"]) > 0.6                    # the synthetic signal is learnable
    f3 = TR.fit("D_FUSION", 16, a, y, w, tr, None, epochs=3, seed=5)
    assert f3["best_epoch"] == 3 and f3["p_val"] is None


def test_stream_padding_does_not_matter_for_e(arrays):
    s, a = arrays
    torch.manual_seed(2)
    m = MD.build("E_COMBINED", 16)
    b1 = TR.tensors(a, np.arange(60))
    b2 = {k: v.clone() for k, v in b1.items()}
    for blk in CF.TV_BLOCKS:
        pad = ~b1[f"sv_{blk}"]
        b2[f"sx_{blk}"][pad] = 4.0; b2[f"sm_{blk}"][pad] = 1.0; b2[f"st_{blk}"][pad] = -2.0; b2[f"sg_{blk}"][pad] = 1.0
    assert np.allclose(_eval(m, b1), _eval(m, b2), atol=1e-5)


def test_stream_times_are_per_assessment_not_per_visit(arrays):
    s, a = arrays
    for blk in CF.TV_BLOCKS:
        v = a[f"sv_{blk}"]
        assert (a[f"st_{blk}"][v] <= 0).all() and (a[f"sg_{blk}"][v] >= 0).all()
    assert (a["dt"][a["valid"]] <= 0).all()
    i = np.arange(s.n)
    assert (a["dt"][i, a["lengths"] - 1] == 0).all()           # anchor token last in the merged view
