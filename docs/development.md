# Development

## Environment

```bash
conda create -n hook python=3.10 -y
conda activate hook
pip install -e .
```

## Package Layout

- Source code is under `src/`.
- Entry points currently run as Python modules from `hook.*`.

## Common Commands

```bash
python -m hook.pipelines.extract_embeddings
python scripts/build_gtzan_metadata.py --split train --out-csv data/metadata/gtzan_train.csv
python -m hook.retrieval.cli --embeddings-pkl artifacts/embeddings/representations.pkl --metadata-csv data/metadata/gtzan_train.csv --genres blues,jazz --k 10
```

## Notes

- `data/`, `artifacts/`, and `archive/` are gitignored by design.
- Retrieval pipeline does not require training.
- Augmentation dependency is lazy-loaded only for contrastive dataset mode.

