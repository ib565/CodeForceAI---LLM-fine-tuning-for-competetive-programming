from datasets import Dataset, load_dataset
import os
import json

DEFAULT_BASE_DIR = "datasets/submissions"
HF_DATASET_PATH = "open-r1/codeforces-submissions"
HF_DATASET_NAME = "selected_accepted"

LANGUAGE_MAP = {
    "C++14 (GCC 6-32)": "cpp",
    "C++17 (GCC 7-32)": "cpp",
    "C++17 (GCC 9-64)": "cpp",
    "C++20 (GCC 11-64)": "cpp",
    "C++20 (GCC 13-64)": "cpp",
    "Clang++17 Diagnostics": "cpp",
    "Clang++20 Diagnostics": "cpp",
    "GNU C++": "cpp",
    "GNU C++0x": "cpp",
    "GNU C++11": "cpp",
    "GNU C++17 Diagnostics": "cpp",
    "MS C++": "cpp",
    "MS C++ 2017": "cpp",
}

KEEP_COLUMNS_V0: set[str] = {
    "submission_id",
    "source",
    "contestId",
    "problem_index",
    "problem_id",
    "programmingLanguage",
    "timeConsumedMillis",
    "memoryConsumedBytes",
}


def _filter_record(record: dict) -> dict:
    filtered: dict = {k: v for k, v in record.items() if k in KEEP_COLUMNS_V0}
    return filtered


def keep_for_v0(p: dict) -> bool:
    if p.get("programmingLanguage") not in LANGUAGE_MAP:
        return False
    return True


def _default_paths_for_split(split: str) -> tuple[str, str, str]:
    base = os.path.join(DEFAULT_BASE_DIR, split)
    os.makedirs(base, exist_ok=True)
    out_jsonl = os.path.join(base, "codeforces_submissions_v0.jsonl")
    ds_dir = os.path.join(base, "codeforces_submissions_v0")
    parquet_path = os.path.join(base, "codeforces_submissions_v0.parquet")
    return out_jsonl, ds_dir, parquet_path


def stream_codeforces_to_jsonl(
    split: str = "train",
    out_jsonl: str | None = None,
    dataset_path: str = HF_DATASET_PATH,
    name: str = HF_DATASET_NAME,
) -> tuple[str, int]:
    if out_jsonl is None:
        out_jsonl, _, _ = _default_paths_for_split(split)

    print(
        f"Streaming {dataset_path} ({name}), split='{split}' " f"-> JSONL: {out_jsonl}"
    )
    stream = load_dataset(
        dataset_path,
        name=name,
        split=split,
        streaming=True,
    )

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in stream:
            if keep_for_v0(ex):
                filtered = _filter_record(ex)
                f.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} filtered records to {out_jsonl}")
    return out_jsonl, count


def build_dataset_from_jsonl(
    out_jsonl: str,
    ds_dir: str | None = None,
    parquet_path: str | None = None,
) -> Dataset:
    if ds_dir is None or parquet_path is None:
        base = os.path.dirname(out_jsonl)
        if ds_dir is None:
            ds_dir = os.path.join(base, "codeforces_submissions_v0")
        if parquet_path is None:
            parquet_path = os.path.join(base, "codeforces_submissions_v0.parquet")

    print(f"Loading dataset directly from JSONL: {out_jsonl}")
    split_name = os.path.basename(os.path.dirname(out_jsonl)) or "train"
    ds_map = load_dataset(
        "json",
        data_files={split_name: out_jsonl},
    )
    ds: Dataset = ds_map[split_name]

    # Ensure only v0 columns remain (defensive in case input JSONL had extras)
    extra_cols = [c for c in ds.column_names if c not in KEEP_COLUMNS_V0]
    if extra_cols:
        ds = ds.remove_columns(extra_cols)

    print("Dataset created")

    os.makedirs(ds_dir, exist_ok=True)
    ds.save_to_disk(ds_dir)
    print(f"Dataset saved to: {ds_dir}")

    ds.to_parquet(parquet_path)
    print(f"Parquet file created at: {parquet_path}")

    return ds


def process_split(split: str = "train") -> None:
    out_jsonl, count = stream_codeforces_to_jsonl(split=split)
    if count == 0:
        print(
            f"Warning: no records passed filtering for split='{split}'. "
            "Skipping dataset build."
        )
        return
    build_dataset_from_jsonl(out_jsonl=out_jsonl)


def process_all_splits(splits: tuple[str, ...] = ("train", "test")) -> None:
    for split in splits:
        try:
            print("=" * 60)
            print(f"Processing split: {split}")
            out_jsonl, _, _ = _default_paths_for_split(split)
            print(f"Output directory: {os.path.dirname(out_jsonl)}")
            process_split(split=split)
        except Exception as e:
            print(
                f"Error processing split '{split}': {e}\n"
                "This split may not exist for the selected dataset/config."
            )


if __name__ == "__main__":
    process_all_splits(("train"))
