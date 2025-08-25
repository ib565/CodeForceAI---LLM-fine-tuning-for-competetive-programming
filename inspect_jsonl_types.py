# inspect_jsonl_types.py
import json
from collections import defaultdict


def inspect(path, n=200):
    types = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            for k, v in rec.items():
                types[k].add(type(v).__name__)
    for k, s in types.items():
        print(k, s)


inspect("datasets/codeforces_v0.jsonl", n=200)
