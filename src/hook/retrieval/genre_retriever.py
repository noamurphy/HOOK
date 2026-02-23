import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def load_embeddings_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Common cases from this repo: Tensor, list[Tensor], or dict payload.
    if hasattr(obj, "detach"):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        if "embeddings" in obj:
            arr = np.asarray(obj["embeddings"])
        elif "representations" in obj:
            arr = np.asarray(obj["representations"])
        else:
            raise ValueError("Dict payload missing 'embeddings'/'representations' keys.")
    elif isinstance(obj, list):
        arr = np.asarray(obj)
    else:
        arr = np.asarray(obj)

    if arr.ndim == 3:
        # Model output can be [N, 512, T]. Pool over time to [N, 512].
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}.")
    return arr.astype(np.float32)


class GenreRetriever:
    def __init__(
        self,
        embeddings: np.ndarray,
        item_ids: Sequence[str],
        item_genres: Sequence[str],
        backend: str = "hnswlib",
    ) -> None:
        if len(embeddings) != len(item_ids) or len(item_ids) != len(item_genres):
            raise ValueError("embeddings, item_ids, and item_genres must have equal length.")

        self.embeddings = _l2_normalize(embeddings.astype(np.float32))
        self.item_ids = list(item_ids)
        self.item_genres = list(item_genres)
        self.backend = backend
        self._centroids = self._compute_genre_centroids()
        self._index = None
        self._build_index()

    def _compute_genre_centroids(self) -> Dict[str, np.ndarray]:
        by_genre: Dict[str, List[np.ndarray]] = {}
        for vec, genre in zip(self.embeddings, self.item_genres):
            by_genre.setdefault(genre, []).append(vec)
        return {genre: _l2_normalize(np.vstack(vectors).mean(axis=0, keepdims=True))[0] for genre, vectors in by_genre.items()}

    def _build_index(self) -> None:
        if self.backend == "hnswlib":
            try:
                import hnswlib  # type: ignore
            except Exception:
                self.backend = "numpy"
                return

            dim = self.embeddings.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(self.embeddings), ef_construction=200, M=16)
            index.add_items(self.embeddings, ids=np.arange(len(self.embeddings)))
            index.set_ef(80)
            self._index = index
            return

        self.backend = "numpy"

    def available_genres(self) -> List[str]:
        return sorted(self._centroids.keys())

    def _target_vector(self, genres: Sequence[str], weights: Optional[Sequence[float]] = None) -> np.ndarray:
        if not genres:
            raise ValueError("At least one genre is required.")
        missing = [g for g in genres if g not in self._centroids]
        if missing:
            raise ValueError(f"Unknown genres: {missing}. Known: {self.available_genres()}")

        if weights is None:
            weights = [1.0] * len(genres)
        if len(weights) != len(genres):
            raise ValueError("weights length must match genres length.")

        w = np.asarray(weights, dtype=np.float32)
        w = w / np.clip(w.sum(), 1e-12, None)
        centroid_stack = np.vstack([self._centroids[g] for g in genres])
        target = (centroid_stack * w[:, None]).sum(axis=0, keepdims=True)
        return _l2_normalize(target)[0]

    def recommend_between(
        self,
        genres: Sequence[str],
        k: int = 10,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[str, str, float]]:
        indexed = self.recommend_between_indexed(genres=genres, k=k, weights=weights)
        return [(self.item_ids[i], self.item_genres[i], score) for i, score in indexed]

    def recommend_between_indexed(
        self,
        genres: Sequence[str],
        k: int = 10,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[int, float]]:
        target = self._target_vector(genres, weights=weights)

        if self.backend == "hnswlib" and self._index is not None:
            labels, distances = self._index.knn_query(target, k=k)
            labels = labels[0]
            distances = distances[0]
            return [(int(i), float(1.0 - d)) for i, d in zip(labels, distances)]

        sims = self.embeddings @ target
        top_idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top_idx]
