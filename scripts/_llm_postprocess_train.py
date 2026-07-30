from pathlib import Path
from _llm_postprocess_common import process

ROOT = Path(__file__).resolve().parent.parent
result = process(
    root=ROOT,
    annotation_dir=ROOT / "results/annotation/train",
    dataset_file=ROOT / "data/primary/train.jsonl",
)
print(result)
