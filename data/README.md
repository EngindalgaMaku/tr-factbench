# TR-FactBench Dataset Files

## Layout

```text
data/
├── primary/
│   └── train.jsonl
├── evaluation/
│   ├── preview_v0.1/
│   │   ├── dev_general.jsonl
│   │   ├── dev_stress.jsonl
│   │   └── domain/          # reviewed domain-specific views for new evaluations
│   └── legacy_experiment_version/
│       ├── dev_general.jsonl
│       ├── dev_stress.jsonl
│       └── domain/          # byte-identical domain subsets used by A1/A3 runs
├── derived/                 # deterministic views of the archived experiment data
├── manifests/               # includes preview manifests and Dev600 ID mapping
├── schema.json
├── example_metadata.csv
├── sources_manifest.csv
├── checksums.sha256
├── DATA_LICENSE.md
└── NOTICE.md
```

`data/primary/` contains training data only. New evaluations should use `data/evaluation/preview_v0.1/`, which contains the completed first blind-human pass. The byte-identical pre-review files used by all archived encoder and LLM experiments are isolated under `data/evaluation/legacy_experiment_version/`. Domain-specific A1/A3 runs resolve to the matching files under its `domain/` subdirectory.

Every archived run manifest points to the legacy directory and retains its originally recorded runtime path under `path_relocation`. File hashes are unchanged. `example_metadata.csv` describes the archived experimental version; the preview JSONL files carry their annotation metadata directly. Dev600 public preview IDs can be joined to archived IDs through `manifests/dev600_legacy_to_preview_id_map.csv`.

Run `python scripts/00_preflight.py` to verify the training and archived experiment inputs, `python scripts/04_build_derived_splits.py` to reconstruct deterministic legacy-derived views, and `python scripts/99_validate_release.py` to validate both layouts and all archived run references.
