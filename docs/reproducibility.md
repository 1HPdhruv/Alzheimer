# Reproducibility

## Public software

Use Python 3.11 and the pinned dependencies. The release package migrates existing project functions into `yoad`; SOURCE_MANIFEST.csv binds each copied or extracted file to its original workspace path/hash. This migration is a new software release, not a replacement historical scientific freeze.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python -m pytest -q -p no:cacheprovider
python scripts/publication_scan.py
python -m yoad.benchmark.runner configs/benchmark/synthetic_smoke_test.yaml
```

The benchmark CLI accepts only its synthetic adapter. Outputs go under `results/benchmark/`, which is ignored. Running from the checkout makes the default output location explicit. The APIs `run(..., results_base=...)` and `run_campaign(..., results_base=...)` allow a chosen output root. Synthetic summaries record seed/configuration identity and software versions. Model-specific seeds are preserved; see configs and implementation. Cross-platform bitwise equality is not guaranteed.

## Restricted scientific reproduction

The public release contains no datasets, folds, private predictions, weights, personal attestations or original frozen input manifests. It cannot reproduce the reported participant-level metrics by itself. Obtain ADNI/NACC access through their respective approved routes, resolve the applicable agreements and institutional approvals, and use a separate researcher-controlled environment. Do not run expensive or participant-level experiments through an AI assistant.

The original raw-data runners and freeze manifests are intentionally excluded: they bind to a private installation, approved source exports and immutable historical dependencies. Do not use new package hashes to claim historical freeze compliance. Reconstruct mappings, cohort/label adjudication, assessment timing, death/etiology policies and participant separation under a newly reviewed version before execution. Record raw export versions privately, code/dependency identities, seeds, tuning, thresholds, missingness rules, all exclusions and output schemas. Have aggregate outputs disclosure-reviewed before public release.

Independent prediction verification is narrower than independent raw-outcome adjudication or full retraining. Report each separately. Conditional prediction bootstraps omit development/selection variability; the full-pipeline bootstrap implementation exists but that does not prove a particular historical run used it.
