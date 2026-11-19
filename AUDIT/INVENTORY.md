# Workspace inventory

Audit dates: 19–20 November 2026 (Asia/Kolkata). Original workspace treated as read-only.

- 14,875 file entries (including symlinks); 1,795,676,217 logical bytes (1.672 GiB).
- 1,829,797,888 allocated bytes summed from file metadata; this is not a unique-block measure for APFS clones.
- Every file entry enumerated; regular-file byte hashes computed locally for duplicates and preservation. No dataset row, private prediction or checkpoint deserialized or printed.
- Inventory errors: 0. No symlink traversal. Hidden files and nested Git/environment files included in totals.
- 98 distinct original files selected; 14,777 excluded. New packaging/docs/tests/audit files are additional release files, not original selections.
- Metadata/hash inspection of all entries is **not** a scientific/content review of all 14,875 files. Content review focuses on release source and status documentation. Unreviewed eligibility defaults to exclusion.

Full original paths and hashes stay in a separate local private audit directory. Public exclusion rows use opaque file IDs to avoid publishing participant identifiers, local account names, correspondence names or restricted file hashes. SOURCE_MANIFEST.csv gives original relative paths and byte identities for released source. LARGE_FILES uses a 10 MiB threshold. Duplicate groups use full byte hashes, not filenames; duplicates have not been removed.

| Category | File entries |
|---|---:|
| other/unresolved | 92 |
| paper/report/figure/notebook | 1406 |
| documentation/literature candidate | 659 |
| table/metadata/provenance candidate | 748 |
| other implementation/configuration | 369 |
| selected source/test/config | 98 |
| environment/cache/history | 10688 |
| archive | 9 |
| symlink | 9 |
| restricted or potentially restricted data/model | 797 |
