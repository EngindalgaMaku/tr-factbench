# Changelog

## v1.2.0 (2026-11, in preparation)

- **Entity masking in the training split.** Names of private companies, brands, and private persons in 47 training context groups (188 instances; contexts, questions, and claims) were replaced with typed aliases local to each group (for example *Banka A*, *Şirket B*, *Portföy Şirketi A*, *Aracı Kurum A*, *Kişi A*), following the masking policy of the held-out set. Labels, identifiers, group structure, and all other text are unchanged. Public authorities, courts, laws, public systems, international organizations, places, and historical figures were not masked.
- The same replacement was applied to every derived or audit file that repeats these instances (`data/derived/`, `results/annotation/train/`, `results/annotation/train_audit_600/`, `results/annotation/deprecated/legacy_iaa_sample_500/`).
- The development sets and the held-out set (Gold-480) are byte-identical to v1.1.0.
- The map from aliases to real names is kept outside the public repository.
- Models and results reported for v1.1.0 were trained on the v1.1.0 training split; a five-seed control run on v1.2.0 is reported in `results/statistics/POST_REVIEW_V12_CONTROL_v1.json`.
- Added a content removal (takedown) policy (`data/NOTICE.md`).

## v1.1.0

- Third-annotator package and sensitivity analysis for the 48 disputed held-out examples.

## v1.0.0

- Citable release with the frozen Gold-480 held-out set and all reported results.
