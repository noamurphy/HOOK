# HOOK
HOOK is an Anti-Recommender Recommender. HOOK fishes for music recommendations from the space between typical music recommendation spaces.

The SampleCNN used to produce musical representations in HOOK is trained by Contrastive Learning for Musical Representation ([CLMR](https://arxiv.org/pdf/2103.09410.pdf)), a [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)-based training technique developed by Janne Spijkervet and John Ashley Burgoyne. The HOOK SampleCNN itself is a modified version of the CLMR pre-trained SampleCNN implementation found [here](https://github.com/Spijkervet/CLMR), and most of its supporting code is derived from the CLMR project.

## Status

- Stage: retrieval MVP (pre-recommender).
- Model strategy: frozen encoder.
- ANN backend: `hnswlib` (with numpy fallback).

## Project layout

- `src/hook/clmr`: CLMR-derived model and dataset code.
- `src/hook/pipelines`: embedding extraction pipeline.
- `src/hook/retrieval`: retrieval MVP (genre-centroid query + ANN).
- `scripts`: helper scripts (for example metadata builders).
- `notebooks/exploration` and `notebooks/experiments`: research notebooks.
- `reports/figures`: curated figures for docs/presentations.
- `data`, `artifacts`, `archive`: local-only folders (gitignored).

## Documentation

- `docs/architecture.md`: system design and component flow.
- `docs/retrieval-mvp.md`: retrieval usage and success criteria.
- `docs/data-sources.md`: dataset strategy and metadata contract.
- `docs/development.md`: environment and daily dev commands.

## Quickstart

1. Install dependencies:

```bash
pip install -e .
```

2. Extract embeddings:

```bash
python -m hook.pipelines.extract_embeddings
```

3. Build GTZAN metadata:

```bash
python scripts/build_gtzan_metadata.py --split train --out-csv data/metadata/gtzan_train.csv
```

4. Run retrieval query:

```bash
python -m hook.retrieval.cli \
  --embeddings-pkl artifacts/embeddings/representations.pkl \
  --metadata-csv data/metadata/gtzan_train.csv \
  --genres blues,jazz \
  --k 10
```

## Immediate Next Steps

1. Run retrieval end-to-end on GTZAN and review qualitative results.
2. Add a larger candidate catalog (~10K) with richer metadata.
3. Persist and benchmark ANN index performance as catalog grows.

## License

This project is licensed under Apache License 2.0. See `LICENSE`.
