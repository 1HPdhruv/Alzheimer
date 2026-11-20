"""Figures drawn only from the AGGREGATE CSVs of a benchmark run folder (no participant data is read).

Forest plots (estimate with its 95% interval, one axis each) for the development nested-CV results of the full-coverage
population, the modality ablation (both frozen models), and binned calibration plots of the locked held-out evaluation. Colour
encodes role only (blue = frozen reference, orange = selected model, grey = other candidates); every point is also labelled on
the axis, so identity never depends on colour. Palette validated with the dataviz validator (light surface, all checks pass).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
REF, WIN, OTHER = "#2a78d6", "#eb6834", "#8a8984"
LABEL = {"brier": "Brier score (lower is better)", "roc_auc": "AUROC (higher is better)"}


def _short(cid: str) -> str:
    s = cid.split("|")
    return f"{s[2]} · {s[3]}" + ("" if s[4] == "median" else f" · {s[4]}") + ("" if s[5] == "primary" else f" · {s[5]}")


def _axes(plt, n_rows, title, xlabel):
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.24 * n_rows), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    return fig, ax


def forest(df: pd.DataFrame, metric: str, title: str, path: Path, ref_id=None, win_id=None, lower_is_better=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=[metric]).sort_values(metric, ascending=not lower_is_better).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = _axes(plt, len(d), title, f"{LABEL.get(metric, metric)}: mean over outer repeats, 95% participant-bootstrap interval")
    for i, r in d.iterrows():
        c = REF if r["candidate"] == ref_id else WIN if r["candidate"] == win_id else OTHER
        lo, hi = r.get(f"{metric}_ci_low"), r.get(f"{metric}_ci_high")
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([lo, hi], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r[metric], i, "o", color=c, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if ref_id is not None and ref_id in set(d["candidate"]):
        ax.axvline(float(d.loc[d["candidate"] == ref_id, metric].iloc[0]), color=REF, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(d))); ax.set_yticklabels([_short(c) for c in d["candidate"]], color=INK)
    ax.set_ylim(-0.7, len(d) - 0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=REF, marker="o", linewidth=2, label="frozen reference (CORE7, L2 logistic)")]
    if win_id is not None:
        handles.append(Line2D([], [], color=WIN, marker="o", linewidth=2, label="selected by the frozen rule"))
    handles.append(Line2D([], [], color=OTHER, marker="o", linewidth=2, label="other candidates"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), fontsize=8, frameon=False,
              labelcolor=INK2)                                   # above the plot: never covers a point
    ax.set_title(title, loc="left", color=INK, fontsize=10, pad=22)
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def calibration(bins: pd.DataFrame, labels: dict, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (3, 3)), label="ideal")
    for key, (name, colour) in labels.items():
        b = bins[(bins["model"] == key) & bins["observed_proportion"].notna()]
        if len(b):
            ax.plot(b["mean_predicted"], b["observed_proportion"], "-o", color=colour, linewidth=2, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8); ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted risk in bin", color=INK2, fontsize=9)
    ax.set_ylabel("observed proportion (bins with suppressed counts omitted)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)


def write_figures(run_dir) -> list:
    run_dir = Path(run_dir)
    out = run_dir / "figures"; out.mkdir(exist_ok=True)
    made = []
    cr = pd.read_csv(run_dir / "candidate_results.csv")
    sel = json.loads((run_dir / "selection.json").read_text()) if (run_dir / "selection.json").exists() else {}
    ref = "A|P_ALL|FS08_CORE7|lr_l2_ref|median|primary"
    win = sel.get("P_ALL", {}).get("winner")
    pool = cr[(cr["population"] == "P_ALL") & cr["stage"].isin(["A", "B"]) & (cr["imputation"] == "median")
              & (cr["timing"] == "primary")]
    for metric, low in (("brier", True), ("roc_auc", False)):
        p = out / f"development_P_ALL_{metric}.png"
        forest(pool, metric, f"Development nested CV, all REF-labelled participants: {LABEL[metric]}", p, ref, win, low)
        made.append(p.name)
    mri = cr[(cr["population"] == "P_MRI") & cr["stage"].isin(["A", "B"]) & (cr["imputation"] == "median")]
    if len(mri):
        p = out / "development_P_MRI_roc_auc.png"
        forest(mri, "roc_auc", "Development nested CV, MRI subset: AUROC", p, ref.replace("P_ALL", "P_MRI"),
               sel.get("P_MRI", {}).get("winner"))
        made.append(p.name)
    hb = run_dir / "heldout_calibration_bins.csv"
    if hb.exists():
        bins = pd.read_csv(hb)
        p = out / "heldout_calibration.png"
        calibration(bins, {"REFERENCE_A3|lr_l2_ref": ("frozen reference (A3, L2 logistic)", REF),
                           "WINNER_P_ALL": ("selected model", WIN)}, "Locked held-out evaluation: calibration", p)
        made.append(p.name)
    return made
