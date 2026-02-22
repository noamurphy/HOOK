# Decisions Log

This file records implementation decisions that affect reproducibility, retrieval behavior, and MVP evaluation.

## 2026-02-22: Inference and Runtime Compatibility Fixes for GTZAN Embedding Extraction

### Decision

Use a deterministic, inference-specific preprocessing path for GTZAN embedding extraction and avoid `torchcodec`-dependent audio decoding.

### Why

Recent `datasets`/`torchaudio` versions in fresh environments attempted to decode audio via `torchcodec`, which failed due to FFmpeg shared library/runtime compatibility on macOS. This blocked embedding extraction for new users even though the project logic itself was correct.

### Changes Made

1. Added GTZAN dataset source fallback and override support:
   - candidate order: `sanchit-gandhi/gtzan`, `marsyas/gtzan`, `ParitKansal/marsyas-gtzan`
   - env override: `HOOK_GTZAN_DATASET_ID`
2. Disabled Hugging Face automatic audio decoding (`Audio(decode=False)`).
3. Loaded audio with `soundfile` instead of `torchaudio.load` in the GTZAN loader.
4. Added fallback to `audio["bytes"]` when dataset `path` fields are invalid/non-local.
5. Converted audio to mono for model compatibility.
6. Added deterministic fixed-length crop/pad to `59049` samples (CLMR input length).
7. Made `SampleCNN.forward` robust to `[B, T]`, `[B, 1, T]`, and `[B, C, T]` inputs.
8. Set embedding extraction `DataLoader(drop_last=False)` so embeddings count matches metadata row count.
9. Added progress logging during embedding extraction.

### Alternatives Considered

1. Install/configure FFmpeg + torchcodec compatibility in every user environment.
2. Pin older `torchaudio`/`datasets` versions to match legacy runtime behavior.
3. Use the contrastive augmentation path for inference.

### Why These Alternatives Were Not Chosen (for MVP)

1. Native codec setup is fragile and hard to document across systems.
2. Pinning old versions increases long-term maintenance burden.
3. Random contrastive augmentations reduce deterministic/repeatable retrieval embeddings.

### Impact on Results

1. Embedding extraction is now portable in fresh environments (especially macOS/conda setups).
2. Inference preprocessing is deterministic, improving repeatability.
3. Results may differ from legacy notebook outputs if old runs used different audio decode stacks or preprocessing defaults.

### Evaluation Notes

When assessing retrieval quality for the MVP, assume:

1. GTZAN audio is mono.
2. Each track is represented by a single deterministic `59049`-sample segment (center-crop/pad).
3. No random contrastive augmentation is applied during inference.
4. All samples are embedded (`drop_last=False`), so metadata and embeddings should match exactly.

### How to Revisit Later

1. Compare single-segment embeddings vs multi-segment averaged embeddings.
2. Add a configurable inference preprocessing policy (crop strategy, segment count).
3. Benchmark `soundfile` vs `torchaudio` paths in a known-good codec environment.
4. Re-evaluate whether `datasets[audio]`/`torchcodec` is worth reintroducing once runtime compatibility is stable.

