# Retrieval MVP

## Goal

Validate that "between genres" retrieval returns musically plausible results from embedding space.

## Inputs

- Embeddings pickle: `artifacts/embeddings/representations.pkl`
- Metadata CSV with `item_id,genre` row-aligned to embeddings

## Commands

1. Extract embeddings:

```bash
python -m hook.pipelines.extract_embeddings
```

2. Build metadata:

```bash
python scripts/build_gtzan_metadata.py --split train --out-csv data/metadata/gtzan_train.csv
```

3. Query retrieval:

```bash
python -m hook.retrieval.cli \
  --embeddings-pkl artifacts/embeddings/representations.pkl \
  --metadata-csv data/metadata/gtzan_train.csv \
  --genres blues,jazz \
  --k 10
```

## Weighted Multi-Genre Query

```bash
python -m hook.retrieval.cli \
  --embeddings-pkl artifacts/embeddings/representations.pkl \
  --metadata-csv data/metadata/gtzan_train.csv \
  --genres blues,jazz,rock \
  --weights 0.5,0.3,0.2 \
  --k 20
```

## Success Criteria

1. Results include tracks from mixed-genre boundaries rather than only one source genre.
2. Retrieval latency remains practical for iterative exploration.
3. Query behavior is stable across repeated runs with same inputs.

