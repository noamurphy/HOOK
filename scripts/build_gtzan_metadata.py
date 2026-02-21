import argparse
import csv

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
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install requirements before running this script."
        ) from exc

    ds = load_dataset("marsyas/gtzan", split=args.split)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "genre", "source_path", "split"])
        writer.writeheader()
        for i, row in enumerate(ds):
            genre = GENRES[row["genre"]]
            file_path = row["file"]
            writer.writerow(
                {
                    "item_id": f"gtzan_{args.split}_{i}",
                    "genre": genre,
                    "source_path": file_path,
                    "split": args.split,
                }
            )

    print(f"Wrote {len(ds)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
