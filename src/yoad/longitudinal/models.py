"""Neural candidates B-E of temporal multimodal v1 (PROTOCOL.md §5). Pure PyTorch modules; no file or data access.

Inputs are the dict produced by sequences.Preprocessor.transform, as tensors:
  merged view (B, C)  x_<block> / m_<block> (N, T, d) values and masks per distinct observation date, valid (N, T), dt (elapsed
                      years, <= 0) and gap (years since the previous date) per token; the last real token is the anchor token
  streams (E)         sx_ / sm_ (N, L, d), st_ (each assessment's own elapsed years), sg_ (gap within the modality), sv_ valid
  last obs (D)        lx_ / lm_ / ls_ (N, d) value, mask, staleness from the assessment date
  static              x_STATIC / m_STATIC (N, d_s) Every value is paired with its mask: a projection
always sees [value, mask], so an unobserved value (stored as 0) is never an unmarked zero. Absent modalities are removed from
attention by key-padding masks or masked softmax, never represented by a learned or zero "measurement".

  B_LSTM         date token = mean of modality projections over the modalities observed on that date + time projection;
                 one-layer LSTM over the real tokens (packed; the anchor token is last); last state + static pathway -> head
  C_TRANSFORMER  same visit tokens + [CLS]; one Transformer encoder layer with key-padding mask; [CLS] + static -> head
  D_FUSION       no temporal model: per-modality encoder of [last value, mask, staleness]; static as a modality; gated
                 attention (masked softmax over present modalities, GateNet-style) with modality dropout in training
  E_COMBINED     one token per dated assessment of each modality stream: projection + modality embedding + projection of
                 its own elapsed time and within-modality gap; static token and [CLS]; one Transformer encoder layer =
                 temporal and cross-modal attention jointly; modality dropout
  E_SNAP         E_COMBINED applied to each modality's most recent assessment only (sequences.latest_only)
"""
from __future__ import annotations

import torch
from torch import nn

from yoad.longitudinal import config as CF

TV = tuple(CF.TV_BLOCKS)
NEG = -1.0e4


def dims() -> dict:
    return {b: len(f) for b, f in CF.TV_BLOCKS.items()} | {"STATIC": sum(len(f) for f in CF.STATIC_BLOCKS.values())}


def present(batch: dict, b: str) -> torch.Tensor:
    """(N, L) 1 where modality b has at least one observed feature at that token."""
    return (batch[f"m_{b}"].sum(-1) > 0).float()


def modality_dropout(batch: dict, p: float, generator: torch.Generator | None, training: bool) -> dict:
    """Training only: drop each time-varying modality of an anchor (all its tokens) with probability p, never all of the
    anchor's observed modalities. Values and masks are zeroed together, so a dropped modality is marked absent."""
    if not training or p <= 0:
        return batch
    n = batch["valid"].shape[0]
    have = torch.stack([present(batch, b).amax(1) for b in TV], 1)                      # (N, M) observed anywhere
    drop = (torch.rand(have.shape, generator=generator) < p).float() * have
    keep_none = (have - drop).sum(1) == 0                                               # would remove everything -> keep all
    drop[keep_none] = 0.0
    out = dict(batch)
    for i, b in enumerate(TV):
        keep = 1.0 - drop[:, i]
        for key in (f"x_{b}", f"m_{b}", f"lx_{b}", f"lm_{b}", f"ls_{b}", f"sx_{b}", f"sm_{b}"):
            if key in batch:
                out[key] = batch[key] * keep.view(n, *([1] * (batch[key].dim() - 1)))
        if f"sv_{b}" in batch:
            out[f"sv_{b}"] = batch[f"sv_{b}"] & (keep > 0.5).view(n, 1)
    return out


class _Common(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        d = dims()
        self.hidden = hidden
        self.proj = nn.ModuleDict({b: nn.Linear(2 * d[b], hidden) for b in TV})
        self.time = nn.Linear(2, hidden)
        self.static = nn.Sequential(nn.Linear(2 * d["STATIC"], hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)

    def static_emb(self, batch):
        return self.static(torch.cat([batch["x_STATIC"], batch["m_STATIC"]], -1))

    def visit_tokens(self, batch):
        """(N, L, H): mean of the projections of the modalities observed at each token, plus the time projection."""
        acc, cnt = 0.0, 0.0
        for b in TV:
            pb = present(batch, b).unsqueeze(-1)
            acc = acc + pb * self.proj[b](torch.cat([batch[f"x_{b}"], batch[f"m_{b}"]], -1))
            cnt = cnt + pb
        t = self.time(torch.stack([batch["dt"], batch["gap"]], -1))
        return acc / cnt.clamp(min=1.0) + t


class LSTMModel(_Common):
    def __init__(self, hidden: int, dropout: float, **_):
        super().__init__(hidden, dropout)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)

    def forward(self, batch, generator=None):
        tok = self.drop(self.visit_tokens(batch))
        lengths = batch["lengths"].clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(tok, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return self.head(torch.cat([h[-1], self.static_emb(batch)], -1)).squeeze(-1)


def _encoder(hidden, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout, batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TemporalTransformer(_Common):
    def __init__(self, hidden: int, dropout: float, heads: int = 2, layers: int = 1, **_):
        super().__init__(hidden, dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.enc = _encoder(hidden, heads, layers, dropout)

    def forward(self, batch, generator=None):
        tok = self.visit_tokens(batch)
        n = tok.shape[0]
        x = torch.cat([self.cls.expand(n, 1, -1), tok], 1)
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool), ~batch["valid"].bool()], 1)
        h = self.enc(x, src_key_padding_mask=pad)[:, 0]
        return self.head(torch.cat([h, self.static_emb(batch)], -1)).squeeze(-1)


class MissingModalityFusion(nn.Module):
    def __init__(self, hidden: int, dropout: float, modality_dropout: float = 0.2, **_):
        super().__init__()
        d = dims()
        self.p_md = modality_dropout
        self.enc = nn.ModuleDict({b: nn.Sequential(nn.Linear(3 * d[b], hidden), nn.ReLU(), nn.Dropout(dropout),
                                                   nn.Linear(hidden, hidden)) for b in TV})
        self.static = nn.Sequential(nn.Linear(2 * d["STATIC"], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.mod_emb = nn.Parameter(torch.zeros(len(TV) + 1, hidden))
        self.gate = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.last_gate_ = None

    def forward(self, batch, generator=None):
        batch = modality_dropout(batch, self.p_md, generator, self.training)
        embs, mask = [], []
        for b in TV:
            embs.append(self.enc[b](torch.cat([batch[f"lx_{b}"], batch[f"lm_{b}"], batch[f"ls_{b}"]], -1)))
            mask.append((batch[f"lm_{b}"].sum(-1) > 0).float())
        embs.append(self.static(torch.cat([batch["x_STATIC"], batch["m_STATIC"]], -1)))
        mask.append(torch.ones_like(mask[0]))                                           # static pathway always present
        E = torch.stack(embs, 1) + self.mod_emb                                          # (N, M + 1, H)
        M = torch.stack(mask, 1)
        g = self.gate(torch.tanh(E)).squeeze(-1).masked_fill(M < 0.5, NEG)
        w = torch.softmax(g, 1) * M
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.last_gate_ = w.detach()
        return self.head((w.unsqueeze(-1) * E).sum(1)).squeeze(-1)


class CombinedTemporalMultimodal(nn.Module):
    def __init__(self, hidden: int, dropout: float, heads: int = 2, layers: int = 1, modality_dropout: float = 0.2, **_):
        super().__init__()
        d = dims()
        self.p_md = modality_dropout
        self.proj = nn.ModuleDict({b: nn.Linear(2 * d[b], hidden) for b in TV})
        self.static = nn.Linear(2 * d["STATIC"], hidden)
        self.time = nn.Linear(2, hidden)
        self.mod_emb = nn.Parameter(torch.zeros(len(TV) + 1, hidden))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.enc = _encoder(hidden, heads, layers, dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.static_head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

    def forward(self, batch, generator=None):
        batch = modality_dropout(batch, self.p_md, generator, self.training)
        n = batch["valid"].shape[0]
        toks, pads = [], []
        for i, b in enumerate(TV):
            t = self.time(torch.stack([batch[f"st_{b}"], batch[f"sg_{b}"]], -1))         # each assessment's own times
            toks.append(self.proj[b](torch.cat([batch[f"sx_{b}"], batch[f"sm_{b}"]], -1)) + t + self.mod_emb[i])
            pads.append(~batch[f"sv_{b}"].bool())                                        # absent / padded -> no token
        s = self.static(torch.cat([batch["x_STATIC"], batch["m_STATIC"]], -1))
        x = torch.cat([self.cls.expand(n, 1, -1), (s + self.mod_emb[-1]).unsqueeze(1)] + toks, 1)
        pad = torch.cat([torch.zeros(n, 2, dtype=torch.bool)] + pads, 1)
        h = self.enc(x, src_key_padding_mask=pad)[:, 0]
        return self.head(torch.cat([h, self.static_head(s)], -1)).squeeze(-1)


ARCH = {"B_LSTM": LSTMModel, "C_TRANSFORMER": TemporalTransformer, "D_FUSION": MissingModalityFusion,
        "E_COMBINED": CombinedTemporalMultimodal, "E_SNAP": CombinedTemporalMultimodal}


def build(model_id: str, hidden: int) -> nn.Module:
    kw = dict(CF.NEURAL_FIXED)
    return ARCH[model_id](hidden=hidden, dropout=kw["dropout"], heads=kw["heads"], layers=kw["layers"],
                          modality_dropout=kw["modality_dropout"])


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
