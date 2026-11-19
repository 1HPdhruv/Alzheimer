"""Guard-railed local runner for the authorised reconciliation checks. NOT executed by the AI assistant on real files.

Guards: explicit confirmation flag; outbound network blocked while checks run; only aggregates written, with small-cell
suppression; a local log with SHA-256 of every input; no participant rows ever printed. Paths come from a local JSON config.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time

import pandas as pd

from yoad.audit import authorized_reconciliation as R

K_DEFAULT = 11


class NetworkBlocked(RuntimeError):
    pass


@contextlib.contextmanager
def block_network():
    """Make any outbound connection or DNS lookup raise while the checks run (defence in depth, not a sandbox)."""
    orig_connect, orig_gai, orig_cex = socket.socket.connect, socket.getaddrinfo, getattr(socket.socket, "connect_ex", None)
    def _deny(*a, **k): raise NetworkBlocked("network access is blocked during local analysis")
    socket.socket.connect = _deny; socket.getaddrinfo = _deny
    if orig_cex: socket.socket.connect_ex = _deny
    try:
        yield
    finally:
        socket.socket.connect, socket.getaddrinfo = orig_connect, orig_gai
        if orig_cex: socket.socket.connect_ex = orig_cex


def suppress_small_cells(obj, k: int = K_DEFAULT):
    """Replace integer counts 1..k-1 by the string '<k' (0 stays 0); recurse through dicts/lists. Floats/bools untouched."""
    if isinstance(obj, bool): return obj
    if isinstance(obj, int): return f"<{k}" if 0 < obj < k else obj
    if isinstance(obj, dict): return {key: suppress_small_cells(v, k) for key, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [suppress_small_cells(v, k) for v in obj]
    return obj


def run_cohort_checks(cfg: dict) -> dict:
    """cfg: {"cohort_csv": path, "columns": {...optional overrides...}, "horizon_days": 730}"""
    cohort = pd.read_csv(cfg["cohort_csv"]); cols = {**R.CHECK_COHORT_COLUMNS, **cfg.get("columns", {})}
    return {"one_row_per_participant": R.one_row_per_participant(cohort, cols["id"]),
            "cohort_counts": R.recompute_cohort_counts(cohort, cols),
            "two_year_rule": R.two_year_eligibility(cohort, cols, cfg.get("horizon_days", 730)),
            "outcome_consistency": R.outcome_censoring_consistency(cohort, cols)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run aggregate reconciliation checks locally (authorised environment only)")
    ap.add_argument("--config", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--authorised-environment-confirmed", action="store_true",
                    help="I confirm this is an approved environment, no AI tool can read these files or outputs, and the applicable agreement permits this analysis")
    ap.add_argument("--min-cell", type=int, default=K_DEFAULT)
    a = ap.parse_args(argv)
    if not a.authorised_environment_confirmed:
        sys.stderr.write("Refusing to run: confirmation flag missing.\n"); return 2
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = json.load(open(a.config)); t0 = time.time()
    hashes = {p: R.file_sha256(p) for p in [cfg["cohort_csv"]]}
    with block_network():
        result = run_cohort_checks(cfg)
    report = {"aggregates": suppress_small_cells(result, a.min_cell), "min_cell": a.min_cell}
    with open(os.path.join(a.out_dir, "report.json"), "w") as f: json.dump(report, f, indent=2)
    with open(os.path.join(a.out_dir, "run_log.txt"), "w") as f:
        f.write(f"run_seconds={time.time() - t0:.1f}\npython={sys.version.split()[0]}\npandas={pd.__version__}\n")
        for p, h in hashes.items(): f.write(f"sha256 {os.path.basename(p)} {h}\n")
    print(f"Wrote aggregate report and log to {a.out_dir} (no rows printed).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
