import argparse
import csv
import os
from pathlib import Path

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build row-aligned GTZAN metadata CSV.")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation"], help="Dataset split")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Optional HF dataset id override. Falls back to HOOK_GTZAN_DATASET_ID or internal candidates.",
    )
    args = parser.parse_args()

    try:
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install requirements before running this script."
        ) from exc

    candidates = []
    if args.dataset_id:
        candidates.append(args.dataset_id)
    env_id = os.environ.get("HOOK_GTZAN_DATASET_ID", "").strip()
    if env_id and env_id not in candidates:
        candidates.append(env_id)
    for ds_id in ["sanchit-gandhi/gtzan", "marsyas/gtzan", "ParitKansal/marsyas-gtzan"]:
        if ds_id not in candidates:
            candidates.append(ds_id)

    ds = None
    loaded_id = None
    errors = []
    for ds_id in candidates:
        try:
            ds = load_dataset(ds_id, split=args.split)
            loaded_id = ds_id
            break
        except Exception as exc:
            errors.append((ds_id, str(exc)))

    if ds is None:
        details = "\n".join([f"- {ds_id}: {msg}" for ds_id, msg in errors])
        raise SystemExit(
            "Failed to load any GTZAN dataset candidate.\n"
            "Use --dataset-id or HOOK_GTZAN_DATASET_ID to set a working source.\n"
            f"Tried:\n{details}"
        )

    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item_id", "genre", "title", "artist", "source_path", "split"],
        )
        writer.writeheader()
        for i, row in enumerate(ds):
            genre = GENRES[row["genre"]]
            if "file" in row and row["file"]:
                file_path = row["file"]
            elif "audio" in row and isinstance(row["audio"], dict):
                file_path = row["audio"].get("path")
            else:
                file_path = ""
            title = Path(file_path).stem if file_path else f"gtzan_{args.split}_{i}"
            writer.writerow(
                {
                    "item_id": f"gtzan_{args.split}_{i}",
                    "genre": genre,
                    "title": title,
                    "artist": "",
                    "source_path": file_path,
                    "split": args.split,
                }
            )

    print(f"Using dataset id: {loaded_id}")
    print(f"Wrote {len(ds)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
