# v0.1.0-preview Release Checklist

## Repository preparation

- [x] DEV960 annotator 1 completed: 960/960
- [x] Dev600 annotator 1 completed: 600/600
- [x] Single-annotator provisional files generated for both evaluation sets
- [x] Original experiment files isolated under `data/evaluation/legacy_experiment_version/`
- [x] All archived encoder and LLM manifests updated to resolvable legacy paths while retaining original recorded paths
- [x] Preview manifests and SHA-256 hashes generated
- [x] README and README_TR updated
- [x] Data card, model card, schema, licensing note, and citation metadata updated
- [x] Annotation status and quality report updated
- [x] Release lock marks `preview_release_ready: true`
- [x] Release validator checks both preview files
- [x] 144 archived runs validated
- [x] LLM baseline tests passed

## External publication actions

- [ ] Create Git tag `v0.1.0-preview`
- [ ] Create the GitHub release using `reports/GITHUB_RELEASE_v0.1.0-preview.md`
- [ ] Attach or publish the repository archive
- [ ] Optionally publish the selected LoRA adapter after a clean-environment load/inference test

## Stable-release work — not a preview blocker

- [ ] Complete annotator 2 for both evaluation sets
- [ ] Compute pre-adjudication agreement
- [ ] Complete adjudication
- [ ] Complete the planned 600-example training quality audit
- [ ] Record stable-release label changes and rerun selected evaluations if needed
