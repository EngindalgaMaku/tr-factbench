# TR-FactBench Data Licensing

## Project-created annotations

Copyright (c) 2026 Engin Dalga

The project-created claims, questions, labels, identifiers, and benchmark
metadata are released under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** license. Attribution must identify TR-FactBench and
link to this repository.

## Context passages

The JSONL files include context passages derived from or excerpted from
third-party Turkish-language sources. Those passages are redistributed as part
of this research preview, but their underlying rights remain source-dependent.
The CC BY 4.0 grant above does **not** override third-party rights in the
context passages.

Before commercial reuse or redistribution, users must review the corresponding
entry in `data/sources_manifest.csv` and verify the source's terms. Records with
`provenance_status=source_adapted_no_item_url` indicate substantially transformed
materials whose item-level source URLs were not retained. These records are not
rights-cleared for commercial redistribution without independent rights review.

## Release status

This is a research-preview dataset. Both evaluation sets include a completed first
blind-human review pass. Training labels remain construction-time candidates. A second
annotation pass and adjudication are planned for a future release.

## Citation

```bibtex
@software{TRFactBench2026,
  author  = {Dalga, Engin},
  title   = {TR-FactBench: A Multi-Domain Benchmark for Source-Grounded
             Factuality Verification in Turkish},
  year    = {2026},
  url     = {https://github.com/EngindalgaMaku/tr-factbench},
  note    = {Research preview; code under MIT, project annotations under CC BY 4.0, contexts source-dependent}
}
```
