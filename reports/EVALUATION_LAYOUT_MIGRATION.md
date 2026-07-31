# Evaluation Layout Migration

The preview release separates training, current preview evaluation, and archived experiment evaluation files:

```text
data/
├── primary/
│   └── train.jsonl
└── evaluation/
    ├── preview_v0.1/
    │   ├── dev_general.jsonl
    │   ├── dev_stress.jsonl
    │   └── domain/
    │       ├── dev960_general_finance.jsonl
    │       ├── dev960_general_legal.jsonl
    │       ├── dev960_general_medical.jsonl
    │       ├── dev600_stress_finance.jsonl
    │       ├── dev600_stress_legal.jsonl
    │       └── dev600_stress_medical.jsonl
    └── legacy_experiment_version/
        ├── dev_general.jsonl
        ├── dev_stress.jsonl
        └── domain/
            ├── dev960_general_finance.jsonl
            ├── dev960_general_legal.jsonl
            ├── dev960_general_medical.jsonl
            ├── dev600_stress_finance.jsonl
            ├── dev600_stress_legal.jsonl
            └── dev600_stress_medical.jsonl
```

The files under `legacy_experiment_version/` are byte-identical to the inputs referenced by archived result hashes. Full-set experiments point to the two top-level files. A1 and A3/LODO experiments point to the matching legacy domain files.

All 144 `dataset_manifest.json` files contain resolvable repository-relative paths. Their original runtime paths are retained under `path_relocation`, so the archive records both the historical path and the current release location.

The release validator confirms for every archived encoder run that:

1. the training and evaluation files exist;
2. evaluation paths remain inside the legacy experiment directory;
3. SHA-256 values match the run manifest;
4. original recorded paths remain available.

Archived LLM manifests use the same legacy full-set paths. New evaluations use the reviewed files under `preview_v0.1/`; domain-specific preview files are deterministic filtered views of those full sets. Dev600 identifiers were normalized for the public preview, with a one-to-one mapping stored in `data/manifests/dev600_legacy_to_preview_id_map.csv`.
