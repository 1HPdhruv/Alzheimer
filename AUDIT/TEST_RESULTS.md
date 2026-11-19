# Release software verification

20 November 2026 (Asia/Kolkata). **369 passed, 0 failed, 0 skipped, 0 errors**, in 208.14 seconds. Pytest emitted 1,807 warnings, principally dependency deprecations, synthetic collinearity and unestimable/small-fixture summary warnings. Warnings were not suppressed. This is class C software evidence, not participant-level scientific validation.

Command (from checkout):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider
```

The suite covers cohort/outcome edge cases, indeterminate follow-up, visit codes, dated and masked features, source-only logistic/XGBoost interfaces, checksum/tamper checks, target-role refusal, participant-grouped isolation, calibration/discrimination, threshold fitting, paired/bootstrap behavior, synthetic transport, snapshot fusion, temporal model invariance, positive controls and publication boundaries. Synthetic whole-pipeline tests also exercise held-out/report schemas and serial/parallel consistency. Extracted tests retain their original assertions; historical tests coupled to excluded private orchestrators were not included.

The baseline environment was Python 3.11.9 with the exact versions in requirements.txt. The verification environment used a separate virtualenv inheriting installed scientific dependencies, with imbalanced-learn installed there. A completely clean dependency installation on every platform was not tested. A standard isolated-build wheel installation succeeded, and the installed `python -m yoad.benchmark.runner configs/benchmark/synthetic_smoke_test.yaml` completed with status `success`, no error and no failed metrics. Its numerical diagnostics are deliberately not represented as study findings. Generated synthetic outputs are ignored and not staged.

An initial import migration failed because broad string replacement also changed a test variable and a config identifier. It was corrected to import-only changes; fixture-extraction constants were restored from source. The reported outcome above is the subsequent complete suite. No failing tests were deleted to obtain it. The source AST comparison confirms the intended changes; see MIGRATION_REVIEW.md.

Editable installation with the pre-existing local build backend generated an invalid path file. Standard wheel installation is the tested/documented route. No global or original research environment was modified.

Not performed: private participant training, raw clinical/timing adjudication, independent prediction/interval verification, original freeze execution, fresh NACC validation or historical-figure rendering. Those activities remain distinct from passing this suite. Local full logs remain in the private audit folder because they include workstation paths.
