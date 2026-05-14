#!/usr/bin/env python3
"""Re-aggregate the published qualitative_medical_*.json per (seed, method, scale).

The default `judge_multiseed.aggregate_gpt4o` pools across seeds. For the 3x5
phase 3 grid we need per-seed-per-method aggregates, which is what this script
produces.

Output: per-seed JSON files like
    aggregated_seed{42,123,456}_medical.json
each containing {method: [{scale, mean_alignment, mean_coherence, ...}]}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualitative", required=True,
                   help="qualitative_*.json with per-generation gpt4o_alignment and gpt4o_coherence")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.qualitative).read_text())
    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    else:
        examples = data

    # Bucket by (seed, method, scale)
    buckets: dict[tuple[int, str, float], dict] = defaultdict(lambda: {"a": [], "c": []})
    for ex in examples:
        if "gpt4o_alignment" not in ex or "gpt4o_coherence" not in ex:
            continue
        seed = ex["seed"]
        cond = ex["condition"]
        scale = ex["scale"]
        # Strip "_a{α}" suffix from the condition to get the method
        m = re.match(r"^(.+?)_a-?\d+(?:\.\d+)?$", cond)
        method = m.group(1) if m else cond
        buckets[(seed, method, scale)]["a"].append(ex["gpt4o_alignment"])
        buckets[(seed, method, scale)]["c"].append(ex["gpt4o_coherence"])

    # Group by seed, then method
    by_seed: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (seed, method, scale), v in buckets.items():
        a = np.array(v["a"], dtype=float)
        c = np.array(v["c"], dtype=float)
        by_seed[seed][method].append({
            "scale": float(scale),
            "mean_alignment": float(a.mean()),
            "std_alignment": float(a.std()),
            "mean_coherence": float(c.mean()),
            "std_coherence": float(c.std()),
            "n": int(a.size),
        })

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, methods in by_seed.items():
        # sort scales within each method
        sorted_methods = {m: sorted(rs, key=lambda r: r["scale"]) for m, rs in methods.items()}
        path = out_dir / f"aggregated_seed{seed}_medical.json"
        path.write_text(json.dumps(sorted_methods, indent=2))
        print(f"  → {path}  (methods: {list(sorted_methods.keys())})")


if __name__ == "__main__":
    sys.exit(main())
