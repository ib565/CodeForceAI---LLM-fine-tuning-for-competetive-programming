import json
from datasets import load_dataset


def keep_for_v0(p):
    if p.get("language") != "cpp":
        return False
    if p.get("interaction_format"):
        return False
    if not p.get("executable", False):
        return False
    if p.get("input_mode") != "stdio":
        return False
    if not p.get("official_tests_complete", False):
        return False
    return True


stream = load_dataset(
    "open-r1/codeforces",
    name="verifiable-prompts",
    split="train",
    streaming=True,
)

out_jsonl = "datasets/codeforces_v0.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as f:
    for ex in stream:
        if keep_for_v0(ex):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
