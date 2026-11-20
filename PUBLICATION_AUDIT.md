# Publication audit

20 November 2026 (Asia/Kolkata). **PASS for the bounded public source release.** Scientific/clinical validation and manuscript-submission clearance remain blocked. This is an ordinary branch/PR release, with no automatic merge.

## Exactly included

Every public path, byte size and SHA-256 is listed in [AUDIT/RELEASE_FILES.csv](AUDIT/RELEASE_FILES.csv). The manifest lists itself with its hash/size omitted to avoid recursive self-hashing. The full path tree is [AUDIT/ARCHITECTURE.md](AUDIT/ARCHITECTURE.md). The only pre-existing remote file is README.md, expanded while retaining its original commit in history.

The source selection uses **98 distinct original workspace files**, yielding multiple release files where a fixture and its tests were extracted separately. Packaging, documentation, audit reports, package initializers and the publication scanner/tests are new engineering artifacts. Existing source functions are mapped by path/hash in SOURCE_MANIFEST.csv. Import/path adaptations and two numerical-annotation removals are documented in MIGRATION_REVIEW.md; no scientific algorithm or model-selection rule was rewritten.

Included functionality: cohort/label construction, date/VISCODE handling, NACC investigation/mapping, feature construction, logistic/XGBoost interfaces, grouped development, calibration/discrimination, snapshot and temporal/multimodal code, robustness/provenance helpers and aggregate plotting functions. Original private raw-data orchestrators and historical immutable freezes are excluded; the public synthetic CLI is tested. No empty scientific model modules or invented research history are presented.

## Exactly excluded and why

**14,777 original file entries are excluded.** FILE_CLASSIFICATION.csv has one decision/reason per original entry; LARGE_FILES.csv covers entries at least 10 MiB, and DUPLICATE_REPORT.csv records byte-identical groups. Excluded paths use opaque IDs publicly; the complete local private inventory maps every ID to its original path/hash. No duplicate was deleted or called useless based on its name.

- All restricted or potentially restricted ADNI/NACC tables, MRI images, private predictions, private folds, checkpoints, learned imputers, model artifacts and derivatives: participant-data/disclosure risk.
- Historical result tables, figures, manuscripts and notebook outputs: file-specific public clearance, embedded-content provenance or rights not established. No historical performance table/figure is republished.
- Correspondence, access/authorization records, attestations and private paths: confidential/governance material. Only bounded status summaries are included.
- Third-party data packages, papers and reference implementation files: redistribution rights not established. Dependencies are installed separately. Local baseline source explicitly records independent reimplementation, not copied unlicensed reference code; those annotations do not license the reference repository.
- Historical raw-data runners, frozen input manifests, absolute-installation orchestration and code outside this tested portable scope: private dependency/provenance coupling or unresolved eligibility. The unused temporal report template with fixed historical reference scores is excluded.
- Environments, caches and nested Git histories: regenerable or unrelated administrative artifacts; original trees/history remain preserved.
- Archives, uncertain files and unselected variants: retained privately, with no claim they are scientifically worthless.

## Checks and evidence

| Check | Outcome |
|---|---|
| Original workspace preservation | Full original byte hashes rechecked: zero content changes; zero added/missing entries; see PRESERVATION.md |
| Existing remote history | main initial commit retained as branch ancestor; no force push or replacement |
| Scientific asset map | All 15 requested components mapped, with source paths, dependencies, I/O, tests, evidence boundaries and missing work |
| Software verification | 369 synthetic tests passed; no failures/skips/errors; warning and environment limits in TEST_RESULTS.md |
| Installation | Standard wheel build/install and installed synthetic CLI succeeded; historical private environment not reused as a result claim |
| Source migration | AST comparison after documented import/path/docstring normalization; unchanged scientific logic |
| Secrets/private paths/identifiers | Scanner checks every release/staged UTF-8 file; no unresolved matches permitted |
| Participant content and restricted datasets | Source-only allowlist; only invented fixtures; all released CSVs are audit metadata, not participant/results tables |
| Known fixture identifiers | ID-shaped literals in tests/synthetic_baseline.py are explicitly hand-built synthetic examples, traced to original tests; no real records |
| Oversized/binary/archive files | 5 MiB public per-file limit; no dataset/model/image/archive types allowed; no Git LFS |
| Ignore rules | Private data/output/model/credential/environment patterns excluded; positive/negative checks run before commit |
| Complete staged set | Every path/byte checked against release manifest; complete staged diff retained locally and checked before commit |
| Figures and numerical research claims | No historical figure/table approved; fixed unverified performance annotations withheld; README contains no project performance metric |
| Licensing | Reserved rights; no permissive grant or ownership certificate invented; unlicensed external reference code/papers not vendored |

The scanner is a heuristic complement to source selection and review, not proof that all possible identifiers or secrets can be recognized. No original participant record was deserialized or used in an experiment. Local hashing for inventory/preservation is not scientific validation. Scientific review here cannot adjudicate raw clinical labels, contractual rights, institutional clearance or independent results.

## Release and remaining gates

Original main: `cd324ca518532ff4b9147fb61a88a09c5e750226`. Dedicated branch: `research/repository-v1`, PR base `main`. Publication actions occur only after the complete staged safety review. The final commit and PR URL are supplied in the delivery rather than embedded into a self-referencing commit.

Independent private prediction/interval verification, raw outcome/timing review, E1 instrument exclusions, E2 history-summary comparator, fresh independent validation, literature refresh and contributor/institutional licensing decisions remain open. Original scientific freezes must not be replaced by this migration's hashes. Public source readiness does not clear the Phase 3 scientific or manuscript-submission gate.
