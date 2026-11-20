"""Training of the neural candidates (PROTOCOL.md §5.3). No file or data access.

Loss: binary cross-entropy weighted by participant weights (sequences.weights: each participant's anchors sum to 1), so a
participant with many visits does not count as many participants. No class weighting or resampling: probabilities stay on the
risk scale and imbalance is handled by the Youden threshold chosen on inner out-of-fold predictions. AdamW, fixed learning rate.
Early stopping only on an inner validation split of the outer-training participants (patience); the final outer-fold refit
uses the median inner best epoch and never sees outer-test anchors.
"""
from __future__ import annotations

import numpy as np
import torch

from yoad.longitudinal import config as CF
from yoad.longitudinal import models as MD


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch.Generator().manual_seed(seed)


def tensors(arrays: dict, idx) -> dict:
    idx = np.asarray(idx)
    out = {}
    for k, v in arrays.items():
        a = v[idx]
        kind = torch.long if k == "lengths" else (torch.bool if (k == "valid" or k.startswith("sv_")) else torch.float32)
        out[k] = torch.as_tensor(a, dtype=kind)
    return out


def _loss(logits, y, w):
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1e-8)


def weighted_log_loss(y, p, w) -> float:
    q = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float); w = np.asarray(w, float)
    return float(-(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / w.sum())


@torch.no_grad()
def predict(model, arrays: dict, idx, batch_size: int = 512) -> np.ndarray:
    model.eval()
    idx = np.asarray(idx)
    out = []
    for i in range(0, len(idx), batch_size):
        out.append(torch.sigmoid(model(tensors(arrays, idx[i:i + batch_size]))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(model_id: str, hidden: int, arrays: dict, y: np.ndarray, w: np.ndarray, tr, va=None, epochs: int | None = None,
        seed: int = 0) -> dict:
    """Early stopping on `va` (weighted log loss) when given; otherwise exactly `epochs` epochs. Returns model, best_epoch,
    validation predictions (at the best epoch) and the epoch history length."""
    fx = CF.NEURAL_FIXED
    g = set_seed(seed)
    model = MD.build(model_id, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=fx["lr"], weight_decay=fx["weight_decay"])
    tr = np.asarray(tr)
    yt, wt = torch.as_tensor(y[tr], dtype=torch.float32), torch.as_tensor(w[tr], dtype=torch.float32)
    max_ep = epochs if epochs is not None else fx["max_epochs"]
    best, best_ep, best_state, best_pva, bad = np.inf, 0, None, None, 0
    for ep in range(1, max_ep + 1):
        model.train()
        order = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), fx["batch_size"]):
            b = order[i:i + fx["batch_size"]].numpy()
            loss = _loss(model(tensors(arrays, tr[b]), generator=g), yt[b], wt[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if va is None:
            continue
        pva = predict(model, arrays, va)
        score = weighted_log_loss(y[va], pva, w[va])
        if score < best - 1e-5:
            best, best_ep, bad, best_pva = score, ep, 0, pva
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= fx["patience"]:
                break
    if va is not None and best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "best_epoch": best_ep if va is not None else max_ep, "p_val": best_pva,
            "val_loss": best if va is not None else np.nan}
