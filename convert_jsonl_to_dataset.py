import json
from datasets import Dataset

out_jsonl = "datasets/codeforces_v0.jsonl"

records = []
with open(out_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} filtered records")

ds = Dataset.from_list(records)  # constructs a proper Arrow schema
print("dataset created")
ds.save_to_disk("datasets/codeforces_v0")  # fast reload with load_from_disk
print("dataset saved")
ds.to_parquet("datasets/codeforces_v0.parquet")  # optional single parquet file
print("parquet file created")
