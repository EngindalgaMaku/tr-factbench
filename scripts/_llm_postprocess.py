from pathlib import Path
from _llm_postprocess_common import process

ROOT = Path(__file__).resolve().parent.parent
result = process(
    root=ROOT,
    annotation_dir=ROOT / "results/annotation/dev600",
    dataset_file=ROOT / "data/evaluation/legacy_experiment_version/dev_stress.jsonl",
)
print(result)
