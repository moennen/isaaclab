"""Merge multiple sequence JSON shards into one file.

Usage:
    python merge_sequences.py shard_00.json shard_25.json shard_50.json shard_75.json \
        --output sequences.json
"""

import argparse
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge sequence JSON shards.")
    parser.add_argument("shards", nargs="+", help="Input shard JSON files")
    parser.add_argument("--output", required=True, help="Output merged JSON file")
    args = parser.parse_args()

    all_sequences = []
    config = None
    gen_args = None

    for path in args.shards:
        with open(path) as f:
            data = json.load(f)
        all_sequences.extend(data["sequences"])
        if config is None:
            config = data.get("config")
            gen_args = data.get("args")

    # Sort by seq_id so the output is ordered
    all_sequences.sort(key=lambda s: s["id"])

    output_data = {
        "version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "args": gen_args,
        "config": config,
        "sequences": all_sequences,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f)

    label_counts = {}
    for seq in all_sequences:
        lbl = seq["label"]
        key = ("reachable" if lbl["reachable"] else "unreachable") + "_" + ("success" if lbl["success"] else "failure")
        label_counts[key] = label_counts.get(key, 0) + 1

    print(f"Merged {len(all_sequences)} sequences → {out_path}")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
