# Research asset map

All 15 requested components are mapped in [SCIENTIFIC_COMPONENTS.csv](SCIENTIFIC_COMPONENTS.csv), including actual original relative paths, APIs, dependencies, inputs/outputs, tests, execution boundary and missing evidence.

## 1 ADNI cohort construction

Original: `YOAD_CORE/src/cohort/cohort.py`; `YOAD_CORE/src/cohort/simkhada_dataset.py`; `YOAD_CORE/multimodal_benchmark/cohort.py`.

yoad.data: diagnosis normalization/earliest MCI and reference-specific construction; yoad.multimodal.cohort: eligible anchors and population frames.

Canonical longitudinal frames/raw source frames -> baselines/anchors/exclusion flow. Dependencies: numpy pandas. Tests: test_cohort; test_baseline_library; multimodal/test_mmb_cohort. Raw clinical adjudication and source-dictionary validation pending; earliest-MCI and protocol-baseline are distinct definitions.

## 2 NACC cohort investigation

Original: `YOAD_CORE/src/cohort/nacc_audit.py`; `YOAD_CORE/src/cohort/nacc_mapper.py`; `YOAD_CORE/experiments/nacc_validation/validation_lib.py`.

yoad.data.nacc_audit/nacc_mapper and yoad.experiments.nacc_validation.validation_lib: onset/age/missing-code mapping, reference labels and frozen transport evaluation.

In-memory visits -> mapped features/labels/audit or transport metrics. Dependencies: numpy pandas scipy scikit-learn; reference library. Tests: test_nacc_audit; test_nacc_library. No independent NACC validation certified; delivered dictionary/death/etiology/access must be reviewed.

## 3 preprocessing

Original: `YOAD_CORE/src/preprocessing/align.py`; `YOAD_CORE/src/benchmark/pipeline.py`.

yoad.preprocessing.align and yoad.benchmark.pipeline: nearest-date alignment; fold-fit imputation/scaling/filtering and FitGuard.

Frames and training indices -> matched rows or fitted transforms. Dependencies: pandas numpy scikit-learn. Tests: test_preprocessing_evaluation; test_benchmark_pipeline_isolation. Generic align_nearest is symmetric; strict builders required for prospective prediction.

## 4 VISCODE and exam dates

Original: `YOAD_CORE/src/cohort/visit_anchor_labels.py`; `YOAD_CORE/experiments/visit_anchor_audit/audit_lib.py`.

yoad.data.visit_anchor_labels and yoad.experiments.visit_anchor_audit.audit_lib: visit-month mapping, harmonization and calendar reference.

VISCODE/examination dates and diagnoses -> anchors/months/reference states. Dependencies: pandas numpy. Tests: test_visit_anchor_labels; test_baseline_library. Visit code is not the actual examination date; conflicts and undated records require explicit policy.

## 5 MCI conversion labels

Original: `YOAD_CORE/src/cohort/reference_outcome.py`; `YOAD_CORE/src/cohort/endpoint.py`.

yoad.data.reference_outcome/endpoint: distinct three-state reference and summary-date endpoint contracts.

Clinical trajectories/summary dates -> EVENT/NONEVENT/INDETERMINATE and reasons. Dependencies: pandas numpy. Tests: test_endpoint_labels; test_reference_outcome_oracle; test_visit_anchor_labels. Clinical source reconstruction not performed; competing death and ascertainment policies are not interchangeable.

## 6 feature engineering

Original: `YOAD_CORE/src/cohort/predictors.py`; `YOAD_CORE/src/cohort/anchor_predictors.py`; `YOAD_CORE/temporal_multimodal/sequences.py`.

yoad.features predictors/anchor_predictors; yoad.longitudinal.sequences: harmonization, dated selection, masks/time and training-only statistics.

Dated assessments and anchors -> feature tables/masked sequences. Dependencies: pandas numpy. Tests: test_preprocessing_evaluation; multimodal/test_mmb_cohort; longitudinal/test_sequences. Code plausibility checks cannot detect every unit/dictionary error; assessment-time review remains required.

## 7 logistic regression

Original: `YOAD_CORE/src/models/frozen_pipeline.py`; `YOAD_CORE/src/models/model_selection.py`.

yoad.models.frozen_pipeline: L2 logistic grouped tuning/imputation/scaling/source calibration and checksum-verified FrozenModel; model_selection: paired source-only comparison.

Named source feature frames/labels/group IDs -> frozen model/probabilities. Dependencies: scikit-learn numpy pandas scipy. Tests: test_frozen_pipeline; test_model_selection; test_freeze_and_leakage. Hyperparameter selection and OOF calibration are not an independent nested performance estimate.

## 8 XGBoost

Original: `YOAD_CORE/src/models/frozen_pipeline.py`; `YOAD_CORE/src/benchmark/models.py`.

yoad.models.frozen_pipeline B_xgboost; yoad.benchmark.models registry; yoad.multimodal.models.

Source features/labels -> probabilities and model state. Dependencies: xgboost scikit-learn numpy. Tests: test_frozen_pipeline; test_freeze_and_leakage; test_benchmark_campaign_and_bootstrap. Comparator status is historical; no verified superiority or clinical readiness.

## 9 multimodal approaches

Original: `YOAD_CORE/multimodal_benchmark/models.py`; `YOAD_CORE/temporal_multimodal/models.py`; `YOAD_CORE/temporal_multimodal/train.py`.

yoad.multimodal.models early/late fusion; yoad.longitudinal.models/train LSTM/Transformer/gated/combined candidates.

Fold-selected feature groups or masked/time tensors -> logits/probabilities. Dependencies: scikit-learn numpy pandas; torch for temporal. Tests: test_benchmark_fusion_and_guards; multimodal/test_mmb_engine; longitudinal/test_models/test_positive_control. Historical GateNet/final-model private orchestrators excluded; no participant-level model comparison independently reproduced.

## 10 calibration

Original: `YOAD_CORE/experiments/today_audit/audit_core.py`; `YOAD_CORE/src/benchmark/metrics.py`; `YOAD_CORE/src/evaluation/frozen_validation.py`.

yoad.experiments.today_audit.audit_core and yoad.benchmark.metrics/frozen_validation: intercept/slope/O:E/Brier and calibration fit.

In-memory labels/probabilities -> calibration summaries/curves. Dependencies: scipy scikit-learn numpy pandas. Tests: test_calibration_library; test_benchmark_config_and_runner; test_frozen_validation. Sparse/one-class inputs may be unestimable; native and updated models must be reported separately.

## 11 discrimination

Original: `YOAD_CORE/src/benchmark/metrics.py`; `YOAD_CORE/multimodal_benchmark/metrics.py`.

yoad.benchmark.metrics and yoad.multimodal.metrics: AUROC/AP, thresholds, cluster/paired uncertainty.

Probabilities, labels, groups -> metrics/intervals/failure status. Dependencies: numpy scipy scikit-learn. Tests: test_calibration_library; multimodal/test_mmb_units; test_benchmark_config_and_runner. Prediction bootstrap is conditional on fitted models, not whole-development uncertainty.

## 12 held-out evaluation

Original: `YOAD_CORE/src/evaluation/safeguards.py`; `YOAD_CORE/multimodal_benchmark/engine.py`; `YOAD_CORE/temporal_multimodal/evaluate.py`.

yoad.evaluation.safeguards; yoad.multimodal.engine; yoad.longitudinal.evaluate: disjoint groups, in-fold operations and held-out APIs.

Grouped frames/folds -> held-out probabilities/summaries. Dependencies: numpy pandas scipy scikit-learn xgboost; torch optional. Tests: test_freeze_and_leakage; test_benchmark_pipeline_isolation; multimodal/test_mmb_engine; longitudinal/test_folds_leakage. Historical held-out reuse is internal; private original folds/thresholds excluded; fresh external evaluation pending.

## 13 diagnostic circularity and robustness

Original: `YOAD_CORE/experiments/visit_anchor_audit/design_lib.py`; `YOAD_CORE/experiments/visit_anchor_audit/discordance_lib.py`; `YOAD_CORE/src/benchmark/design_bootstrap.py`.

yoad.experiments.visit_anchor_audit: 2x2 label/anchor design, predictor sensitivity and discordance; yoad.benchmark.design_bootstrap: refitting uncertainty.

Common grouped cohorts/label choices -> paired contrasts and leakage checks. Dependencies: numpy pandas scipy scikit-learn. Tests: multimodal/test_mmb_cohort/test_mmb_engine; longitudinal/test_folds_leakage. Individual FAQ/CDR/MMSE+ADAS refitted exclusions and history-summary comparator are not certified completed.

## 14 provenance and verification

Original: `YOAD_CORE/src/models/freeze.py`; `YOAD_CORE/src/models/persist.py`; `YOAD_CORE/src/common/run_provenance.py`.

yoad.models.freeze/persist and yoad.common.run_provenance: manifest checksums, tamper checks and software/config/run identity.

Model/config metadata -> hashes/manifests; private writer remains private. Dependencies: Python standard library numpy pandas scikit-learn. Tests: test_freeze_and_leakage; test_frozen_pipeline. Historical original freeze manifests excluded; no independent clinical/adjudication proof. Installed-package provenance helper uses historical relative root logic; do not rely on it for release Git identity.

## 15 visualization

Original: `YOAD_CORE/multimodal_benchmark/figures.py`.

yoad.visualization.figures: forest/calibration plots from aggregate tables.

Reviewed aggregate CSV/frame -> image files. Dependencies: pandas matplotlib. Tests: No original visualization unit suite selected; source syntax/import checked. Not rendered on historical outputs in this release; no figure provenance/disclosure approval inferred.

## Release selection boundaries

Papers/reports and results exist across documentation, reconstruction, both nested repositories and output/tmp generations. Their names do not establish validity or uselessness. Environments/caches are regenerable; archives remain preserved and unexpanded for public release. Notebooks/figures/tables and private freeze artifacts remain withheld. The portable library reuses YOAD_CORE working-tree code; sibling YOAD-Research and Urop_RECOVERY remain historical sources, not silently combined implementations. No separate raw MRI processing or clinically adjudicated onset cohort is manufactured.
