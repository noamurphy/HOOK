# Retrieval MVP

This folder scaffolds a first "between genres" retriever:

1. Load track embeddings.
2. Compute genre centroids.
3. Build an ANN index (`hnswlib`) or fallback to exact cosine search (`numpy`).
4. Query nearest tracks to a weighted genre centroid.

## Inputs

- `embeddings.pkl`: 2D `[N, D]` or 3D `[N, D, T]` tensors/arrays.
- `metadata.csv`: row-aligned with embeddings and with columns:
  - `item_id`
  - `genre`

## Run

```bash
python -m hook.retrieval.cli \
  --embeddings-pkl /path/to/embeddings.pkl \
  --metadata-csv /path/to/metadata.csv \
  --genres blues,jazz \
  --k 10
```

Generate GTZAN-aligned metadata (for quick local testing):

```bash
python scripts/build_gtzan_metadata.py \
  --split train \
  --out-csv /path/to/metadata.csv
```

Optional weights:

```bash
python -m hook.retrieval.cli \
  --embeddings-pkl /path/to/embeddings.pkl \
  --metadata-csv /path/to/metadata.csv \
  --genres blues,jazz,rock \
  --weights 0.5,0.3,0.2 \
  --k 20
```

## Notes

- If your model outputs `[N, 512, T]`, embeddings are mean-pooled to `[N, 512]`.
- `hnswlib` is optional. Without it, retrieval uses exact cosine search.
- Freeze the encoder for MVP; no retraining is required to validate retrieval behavior.
- M-series/Colab training history should not block this phase, since this pipeline is inference + indexing.
