# Known Issues and Limitations

This file documents known technical limitations of the current codebase. Items are numbered to match the audit that identified them.

---

## #14 — `LABEL_MAP` used before assignment inside `tokenize_dataset` (resolved)

**Status:** Fixed.

`src/train.py` defined `tokenize_dataset()` at line ~505 and assigned `LABEL_MAP = LABEL2ID` at line ~533, after the function definition. The function referenced `LABEL_MAP` inside its body. This worked only because of Python's closure late-binding (the variable was resolved at call time, not at definition time), making the code fragile and misleading. Any reordering of the surrounding block would produce an `UnboundLocalError`. The assignment has been moved to immediately before the `tokenize_dataset` definition so the binding order matches the reading order.

---

## #15 — `shuffled_context_claim` used training seed instead of fixed seed=0; no self-match guard (resolved)

**Status:** Fixed.

`apply_shortcut_ablation()` in `src/train.py` used `random.Random(seed)` (the training seed argument) for shuffling contexts in the `shuffled_context_claim` view. The published ablation protocol specifies *"a fixed seed of 0"*; using the training seed produced a different shuffle for every experiment seed, making cross-run comparisons unreliable. Additionally, the plain `rng.shuffle()` call offered no guarantee that a record would not receive its own original context. The view now always uses `random.Random(0)` and applies a single-swap repair pass to eliminate any remaining self-matches after shuffling.

---

## #16 — `hf_release/requirements.txt` minimum versions were incompatible with the training environment (resolved)

**Status:** Fixed.

`hf_release/requirements.txt` required `transformers>=4.40.0` and `peft>=0.10.0`. The archived training environment uses `transformers 5.12.1` and `peft 0.19.1`. The PEFT `SEQ_CLS`/`FEATURE_EXTRACTION` task-type API used by this project is not available in older PEFT versions, and the `all-linear` LoRA target expansion and `exclude_modules` behaviour differ across major versions. Loading the adapter with incompatible versions could silently produce wrong results. The lower bounds have been raised to `transformers>=5.0.0` and `peft>=0.15.0`.

---

## #1 — Default label set in `src/train.py` was `5class` (resolved)

**Status:** Fixed in commit `1025fb9`, updated domain path in `1025fb9` follow-up.

`src/train.py` previously defaulted to `--label_set 5class` (legacy five-class format). The current primary release is four-class. The default is now `4class`. Training defaults use `data/primary/` and `data/derived/train_domain/`. New general, stress, and domain-specific evaluations resolve to `data/evaluation/preview_v0.1/`; archived experiment runners and manifests use `data/evaluation/legacy_experiment_version/`. The direct-training test-path resolver was also corrected. Wrapper `scripts/10_run_experiment.py` always passes `--label_set 4class` and explicit file paths, so archived experiment results are unaffected.

---

## #2 — Output directory does not include full configuration in path

**Status:** Known, not yet addressed.

`src/train.py` writes outputs to `{output_dir}/{model_basename}/{domain}/seed_{seed}/`. If the same model, domain, and seed are used with different hyperparameters (learning rate, LoRA rank, dropout), the previous output is silently overwritten. Wrapper `scripts/10_run_experiment.py` uses a unique `run_id`-based output path, which avoids this problem for all archived runs. Direct `src/train.py` usage is at risk. A future fix should incorporate a configuration hash or `run_id` into the output path.

---

## #3 — Model revision is recorded but not used at load time

**Status:** Known, not yet addressed.

`scripts/10_run_experiment.py` records the Hugging Face commit hash in `model_manifest.json`. However, `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` do not pass `revision=<commit_hash>` at load time. If a model's `main` branch is updated, future runs will use a different revision than the archived experiments. For strict reproducibility, the pinned `revision` should be added to all load calls.

---

## #4 — Trainable parameter manifest uses a second model instance (partially addressed)

**Status:** Known, not yet addressed.

`scripts/10_run_experiment.py` constructs a separate model and LoRA config to count trainable parameters before training starts. This separate instance uses `FEATURE_EXTRACTION` task type uniformly, while `src/train.py` uses `SEQ_CLS` for all non-DistilBERT models. Parameter counts in `trainable_parameters.json` should therefore be treated as estimates. The parameter names may also differ from those in the actual training model. A correct fix requires extracting counts from the real trained model after training completes.

---

## #5 — Peak VRAM path mismatch (resolved)

**Status:** Fixed in commit `1025fb9`. Historical CSV values have not been backfilled.

`src/train.py` writes `peak_memory.json` inside the nested `experiment_dir` (model/domain/seed), not directly into `ckpt/`. The wrapper searched only `ckpt/peak_memory.json` and missed it, leaving `peak_vram_mb` empty for most rows in `results/csv/runs.csv`. The wrapper now uses `rglob("peak_memory.json")` to find the file regardless of nesting depth. Previously collected run data in `runs.csv` was not backfilled; new runs will populate the column correctly.

---

## #6 — Checkpoint selection used lexicographic sort (resolved)

**Status:** Fixed in commit `1025fb9`.

`trainer_state.json` candidates were sorted alphabetically, which places `checkpoint-900` after `checkpoint-1000` in string order. The wrapper now extracts the integer step number from the checkpoint directory name and selects the highest. The fix applies to `scripts/10_run_experiment.py`.

---

## #7 — Unknown label silently mapped to `supported` (resolved in main and eval scripts)

**Status:** Fixed in commit `1025fb9` for `scripts/10_run_experiment.py`. Extended to evaluation and analysis scripts in the follow-up commit.

The following scripts previously used `LABEL2ID.get(label, 0)`, silently treating any unrecognised label as `supported` (id=0):
- `scripts/10_run_experiment.py`
- `scripts/11_reeval_run.py`
- `scripts/20_cross_domain_eval.py`
- `scripts/27_reeval_8sets.py`
- `scripts/30_collapse_analysis.py`
- `scripts/32_error_analysis.py`

All six files now raise a `ValueError` with the offending record and label name on any label not in `LABEL2ID`. Other helper scripts not listed here have not been audited for the same pattern.

---

## #8 — Config files do not capture full experiment definition

**Status:** Known, not yet addressed.

`config.json` in each run directory records the main hyperparameters but omits: model revision SHA, optimizer and scheduler details, gradient accumulation steps, FP16/BF16 selection, exact command-line argv, and package versions. These are partially available in `dataset_manifest.json` and `model_manifest.json`. A future `resolved_config.json` should consolidate all of these into a single document per run.

---

## #9 — `requirements.txt` is not a fully frozen environment

**Status:** Known; confirmed archived versions documented in `requirements.txt` comments.

The `requirements.txt` minimum constraints use `>=`. The confirmed archived experiment environment (from `model_manifest.json`) is: Python 3.14.3, PyTorch 2.11.0+cu126, CUDA 12.6, Transformers 5.12.1, PEFT 0.19.1, scikit-learn 1.8.0. Versions of numpy, scipy, pandas, datasets, accelerate, evaluate, and matplotlib were not recorded in the manifests and are unknown. `requirements-locked.txt` provides closer pinning but still uses `>=` for several packages. A `pip freeze` snapshot from the training environment would be more reliable.

---

## #10 — Selected adapter checkpoint not distributed

**Status:** Known; by design for this preview release.

The model adapter weights are not included in the repository due to file size. The repository is a result archive — metrics, predictions, and configurations are fully reproducible from the run directories, but loading the selected `H1_DROP10_S45` adapter for inference requires re-running the training. A future release may add the adapter via Git LFS or Hugging Face Hub.

---

## #11 — Empty cross-domain placeholder table (resolved)

**Status:** Resolved in the preview release cleanup.

The empty seed-42 placeholder files `table_E_cross_domain_matrix.csv` and `.md` were removed. The populated three-seed summaries remain available as `table_E_cross_domain_dev960_3seed.csv` and `table_E_cross_domain_dev600_3seed.csv`. Release validation now rejects reintroduction of the empty placeholders.

---

## #12 — UTF-8 BOM normalization (resolved)

**Status:** Resolved in the preview release cleanup.

Text artifacts in the release were normalized to plain UTF-8 without BOM. Release validation checks JSON, JSONL, CSV, Markdown, Python, YAML, CFF, and text files for a UTF-8 BOM. Readers continue to accept `utf-8-sig` where compatibility with older snapshots is useful.

---

## #13 — Preview regeneration and Dev600 identifier mapping (resolved)

**Status:** Resolved in the preview release cleanup.

`scripts/43_build_single_annotator_preview.py` now rebuilds both evaluation files from the archived inputs and annotator-1 records. It applies the documented Dev600 claim correction explicitly, regenerates domain-specific preview views, and writes `data/manifests/dev600_legacy_to_preview_id_map.csv` for cross-version joins.
