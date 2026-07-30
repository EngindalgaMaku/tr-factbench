# Legacy experiment evaluation files

These files are the byte-identical evaluation inputs used by the archived encoder and LLM experiments. They are retained only to reproduce the published preview results.

- `dev_general.jsonl`: SHA-256 `2f05485670066558f3884196de110154acd9cc5cfe9be81ddd4f0253effb7f48`
- `dev_stress.jsonl`: SHA-256 `09b277c01f316921bc33b9cc72c8674b929a7019d1a3ad9ac2c45bef663a1658`

For new evaluations, use the files under `../preview_v0.1/`.

The `domain/` subdirectory contains the exact CRLF-serialized 320-example and 200-example domain subsets used by the A1 and A3/LODO runs. Their hashes match the corresponding run manifests.
