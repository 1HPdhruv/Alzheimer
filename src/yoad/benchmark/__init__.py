"""Configurable benchmark framework (Stage 1, 2026-10-17).

Synthetic-tested only. Stage 1 has no real-data adapter: the runner refuses any dataset that is not marked synthetic.
Modules:
  config    - YAML configuration schema and validation (rejects unknown keys, planned components, unsafe outputs)
  data      - DatasetBundle interface, adapter registry, deterministic synthetic generator
  splitting - participant/group-aware holdout and k-fold splits with explicit failure (no silent row-split fallback)
  pipeline  - scikit-learn pipeline builder; all learned steps and resampling live inside the fitted pipeline
  models    - model registry (implemented / optional / planned)
  metrics   - task-aware metric registry (probability vs threshold metrics), calibration and threshold selection
  runner    - experiment runner and structured logging
"""
