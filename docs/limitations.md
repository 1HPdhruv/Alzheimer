# Limitations

- No independent participant-level scientific result was reproduced in this audit. Historical reports disagree in date and scope; the evidence ledger preserves that distinction.
- YOAD is the project name. Enrollment age below a cutoff does not establish young-onset disease; onset-defined feasibility remains distinct from general-MCI progression.
- Diagnosis-linked MMSE, FAQ and CDR content raises incorporation/construct-overlap concerns. Whole-block removal cannot substitute for individual refitted exclusions or establish biological specificity.
- Missing follow-up must not become a stable label. Diagnosis conflicts, etiology, competing death and ascertainment intervals remain clinically consequential and require raw-source review.
- Calendar-month and visit-code outcomes are different estimands. Historical helpers include different endpoint contracts and must not be interchanged.
- `align_nearest` allows a symmetric date window; use strict pre-anchor builders for prospective prediction. Passing generic alignment tests does not establish temporal validity.
- The early frozen logistic/XGBoost interface tunes on grouped source CV; its source calibration reuses selected hyperparameters. This does not provide an independent estimate of complete model-development performance. Use nested designs where required.
- Grouping prevents participant overlap only if identifiers and mappings are correct. Split helpers aggregate participant labels; repeated time-varying outcomes need explicit estimand review.
- In-range unit rescaling may evade plausibility checks. Software checks do not adjudicate clinical correctness or verify all releases of a data dictionary.
- Historical held-out ADNI data were reused according to later manuscript audits. Reuse is internal evaluation, not external confirmation. Conditional intervals omit tuning/training/selection uncertainty.
- Source-only code is portable; historical orchestrators, input manifests and private environments are not included. Full scientific reproducibility remains blocked.
- Publicly available source has reserved rights, not an open-source license. Reference implementation code and papers are not vendored. Literature coverage is not a new systematic review or a novelty guarantee.

This is research software, not a clinically validated diagnostic or treatment system.
