import argparse
import csv
from typing import Dict, List

from .genre_retriever import GenreRetriever, load_embeddings_file


def _parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP genre-bridge retrieval")
    parser.add_argument("--embeddings-pkl", required=True, help="Path to embeddings/representations pickle file")
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="CSV with columns: item_id,genre (same row order as embeddings)",
    )
    parser.add_argument("--genres", required=True, help="Comma-separated input genres, e.g. blues,jazz")
    parser.add_argument("--weights", default=None, help="Optional comma-separated weights matching --genres")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--backend",
        default="hnswlib",
        choices=["hnswlib", "numpy"],
        help="ANN backend (falls back to numpy if hnswlib is not installed)",
    )
    args = parser.parse_args()

    embeddings = load_embeddings_file(args.embeddings_pkl)
    item_ids = []
    item_genres = []
    metadata_rows: List[Dict[str, str]] = []
    with open(args.metadata_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_rows.append(row)
            item_ids.append(row["item_id"])
            item_genres.append(row["genre"])

    retriever = GenreRetriever(
        embeddings=embeddings,
        item_ids=item_ids,
        item_genres=item_genres,
        backend=args.backend,
    )

    genres = _parse_csv_list(args.genres)
    weights = None
    if args.weights:
        weights = [float(x) for x in _parse_csv_list(args.weights)]

    results = retriever.recommend_between_indexed(genres=genres, k=args.k, weights=weights)

    print(f"Query genres: {genres}")
    print(f"Available genres: {retriever.available_genres()}")
    print("Top results:")
    for idx, score in results:
        row = metadata_rows[idx]
        item_id = row.get("item_id", item_ids[idx])
        genre = row.get("genre", item_genres[idx])
        title = row.get("title", "")
        artist = row.get("artist", "")
        source_path = row.get("source_path", "")

        parts = [f"item_id={item_id}", f"genre={genre}", f"similarity={score:.4f}"]
        if title:
            parts.append(f"title={title}")
        if artist:
            parts.append(f"artist={artist}")
        if source_path:
            parts.append(f"source_path={source_path}")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
