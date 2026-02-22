# Architecture

## Purpose

HOOK is a retrieval-first music discovery system.  
It uses frozen CLMR embeddings to find tracks near the "in-between" space of selected genres.

## Current System

1. `hook.pipelines.extract_embeddings` generates embeddings from audio.
2. `scripts/build_gtzan_metadata.py` creates row-aligned metadata for embeddings.
3. `hook.retrieval.cli` builds a query vector from genre centroids and retrieves nearest tracks.

## Core Components

- `src/hook/clmr`: CLMR-derived model and dataset code.
- `src/hook/pipelines/extract_embeddings.py`: embedding extraction entrypoint.
- `src/hook/retrieval/genre_retriever.py`: centroid computation + ANN/exact retrieval.
- `src/hook/retrieval/cli.py`: command-line interface for genre-bridge queries.

## Data and Artifact Flow

1. Input audio dataset is loaded (currently GTZAN for MVP/reference geometry).
2. Encoder produces representations.
3. Representations are saved to `artifacts/embeddings/representations.pkl`.
4. Metadata CSV is generated in `data/metadata/`.
5. Retriever loads embeddings + metadata, computes centroids, and returns top-k matches.

## Query Model

1. Compute one centroid vector per genre from catalog embeddings.
2. Build a target vector from selected genres.
3. Optional weighting supports 3+ genre blends.
4. Retrieve nearest neighbors by cosine similarity.

## Indexing Backends

- Default: `hnswlib` for fast approximate nearest-neighbor search.
- Fallback: `numpy` exact cosine search when ANN dependency is unavailable.

## Design Decisions

- Frozen encoder for MVP: reduces risk and enables faster product iteration.
- Retrieval-first architecture before recommender ranking/personalization.
- Lazy import of augmentation package so non-contrastive paths are less brittle.
- Runtime and inference compatibility decisions are tracked in `docs/decisions.md`.

## Next Architecture Milestones

1. Replace GTZAN-only retrieval corpus with a production candidate catalog.
2. Persist ANN index to disk in `artifacts/indexes/`.
3. Add evaluation harness (recall@k, qualitative "between genre" judgments).
4. Introduce reranking and eventually personalized recommendation logic.
