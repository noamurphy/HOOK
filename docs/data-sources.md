# Data Sources

## Current

- GTZAN (`marsyas/gtzan` via Hugging Face datasets)
  - Role: reference geometry and MVP validation dataset.
  - Not intended as final recommendation catalog.

## Planned Catalog Strategy

1. Start with moderate catalog size (around 10K tracks) for fast iteration.
2. Use legally accessible preview/full-track sources where terms allow embedding generation.
3. Expand gradually while monitoring retrieval quality and indexing performance.

## Metadata Contract

Every embedding row should map to a metadata row with at least:

- `item_id` (stable unique ID)
- `genre` (initial anchor label)

Recommended additional fields:

- `title`
- `artist`
- `source_path` or `source_url`
- `duration_sec`
- `license`

## File Locations

- Raw media: `data/raw/`
- Processed media: `data/processed/`
- Metadata: `data/metadata/`
- Embeddings: `artifacts/embeddings/`
- ANN indexes: `artifacts/indexes/`

