#!/usr/bin/env python3
"""Phase 3 comparison plot: the published QK→OV vs same-budget SAE steering at 4 hookpoints.

What we want to see:
  1. Does the published medical QK→OV result replicate? (top row, leftmost cell)
  2. How does it compare to conventional SAE-feature steering at the four
     neighbouring hookpoints?

Inputs (after judge_multiseed.py has been run on each):
  --orig-medical PATH    a `gpt4o_aggregated_*.json` for medical from the FRA frontier sweep
                         (typically /workspace/runs/medical/gpt4o_aggregated_qualitative_medical_*.json)
  --sae <hook> PATH      one or more "<hookpoint label>=<gpt4o_aggregated json>" pairs
                         e.g. --sae resid_pre=/workspace/runs/phase3.../sae_resid_pre/gpt4o_aggregated_*.json

Output:
  <out>.png  <out>.pdf   side-by-side bars (Δalign|coh≥70 and peak align)
  <out>.json             machine-readable summary

Shape:
  - X axis = method (5 entries: the original QK→OV, then 4 SAE-resid hookpoints)
  - 2 panels: Δalign|coh≥70  and  peak alignment
  - Per-seed colour groups; seed=42 plotted first, additional seeds
    layered as side-by-side bars.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


def _delta_at_floor(rows):
    """Δalign|coh≥70  AND  peak alignment, given a list of {scale, mean_alignment, mean_coherence}."""
    rows = sorted(rows, key=lambda r: r["scale"])
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    mask = co >= COH_FLOOR
    if mask.any():
        delta = float(al[mask].max() - al[mask].min())
        n70 = int(mask.sum())
    else:
        delta = float("nan")
        n70 = 0
    peak = float(np.nanmax(al)) if al.size else float("nan")
    return delta, peak, n70


def load_method_rows(path: Path, prefer_method: str | None = None) -> dict:
    """Load a `gpt4o_aggregated_*.json` and return {method: rows}.

    The judge writes per-method rows under the top-level dict (no `aggregated` key).
    `prefer_method` lets the caller pick e.g. `qk_to_ov` from a multi-method file;
    otherwise return everything.
    """
    data = json.loads(path.read_text())
    # support nested {"aggregated": {...}} schema too
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data = data["aggregated"]
    if prefer_method:
        rows = data.get(prefer_method, [])
        return {prefer_method: rows} if rows else {}
    return data


def parse_sae_arg(s: str):
    """Parse '<label>=<path>' or '<label>:<path>' or '<label>=<path>=<seed>'."""
    if "=" in s:
        parts = s.split("=", 2)
    elif ":" in s:
        parts = s.split(":", 2)
    else:
        raise SystemExit(f"--sae arg '{s}' must be label=path[=seed]")
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], int(parts[2])


def _seed_from_filename(p: str) -> int | None:
    """Best-effort: read seed from filename like *_seed42_*."""
    m = re.search(r"seed[_-]?(\d+)", p)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-medical", required=True,
                   help="gpt4o_aggregated_*.json for medical (FRA frontier sweep)")
    p.add_argument("--orig-method", default="qk_to_ov",
                   choices=["qk_to_ov", "ov_to_ov", "qk_to_qk"])
    p.add_argument("--sae", action="append", default=[], required=True,
                   help="<label>=<path>[=<seed>]; pass once per hookpoint+seed.")
    p.add_argument("--out", required=True,
                   help="Output base path (no extension)")
    p.add_argument("--title", default="Phase 3 — QK→OV vs same-budget SAE steering (medical EM)")
    args = p.parse_args()

    # ── 1. Load the published QK→OV row (the headline) ─────────────────────────
    orig_data = load_method_rows(Path(args.orig_medical), prefer_method=args.orig_method)
    if not orig_data.get(args.orig_method):
        print(f"[FATAL] {args.orig_medical} has no rows for method '{args.orig_method}'")
        return 1
    orig_delta, orig_peak, orig_n70 = _delta_at_floor(orig_data[args.orig_method])
    orig_seed = _seed_from_filename(args.orig_medical) or 0

    # ── 2. Load each SAE-hookpoint result(s) ────────────────────────────
    bars: dict[str, list[dict]] = defaultdict(list)  # method-label → list of {seed, delta, peak, n70}
    bars[f"FRA QK→OV\n(L24 ln1, the original)"].append({
        "seed": orig_seed, "delta": orig_delta, "peak": orig_peak, "n70": orig_n70,
    })
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        d = load_method_rows(Path(path))
        # collapse all method entries (the SAE eval writes a single 'sae_resid' method)
        rows = []
        for m, r in d.items():
            rows.extend(r)
        delta, peak, n70 = _delta_at_floor(rows)
        bars[f"SAE\n{label}"].append({"seed": seed, "delta": delta, "peak": peak, "n70": n70})

    # ── 3. Plot ─────────────────────────────────────────────────────────
    methods = list(bars.keys())  # preserve insertion order
    seeds = sorted({b["seed"] for entries in bars.values() for b in entries})
    n_seeds = max(1, len(seeds))
    width = 0.8 / n_seeds

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("viridis")
    seed_colours = {s: cmap(i / max(1, n_seeds - 1)) for i, s in enumerate(seeds)} \
                   if n_seeds > 1 else {seeds[0]: "#58a6ff"}

    x = np.arange(len(methods))

    for panel, key, label in [(axes[0], "delta", f"Δalign | coh≥{int(COH_FLOOR)}"),
                              (axes[1], "peak",  "peak alignment")]:
        for si, seed in enumerate(seeds):
            heights = []
            for m in methods:
                vals = [b[key] for b in bars[m] if b["seed"] == seed]
                heights.append(vals[0] if vals else np.nan)
            offset = (si - (n_seeds - 1) / 2) * width
            panel.bar(x + offset, heights, width=width, color=seed_colours[seed],
                      edgecolor="black", linewidth=0.5, label=f"seed={seed}")
            for xi, h in enumerate(heights):
                if not np.isnan(h):
                    panel.text(x[xi] + offset, h + 0.5, f"{h:.1f}",
                               ha="center", va="bottom", fontsize=9)
        panel.set_xticks(x)
        panel.set_xticklabels(methods, fontsize=10)
        panel.set_ylabel(label)
        panel.set_title(label)
        panel.grid(axis="y", linestyle=":", alpha=0.4)
        panel.axhline(0, color="black", lw=0.5)
        if n_seeds > 1:
            panel.legend(loc="upper right", fontsize=9)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()

    out_base = Path(args.out).expanduser()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=160)
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)

    # ── 4. JSON summary ─────────────────────────────────────────────────
    summary = {
        "coh_floor": COH_FLOOR,
        "orig_method": args.orig_method,
        "methods": {m: bars[m] for m in methods},
    }
    json_path = Path(str(out_base) + ".json")
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Phase 3 comparison ===")
    for m, entries in bars.items():
        for e in entries:
            print(f"  {m:30s}  seed={e['seed']:>3}  Δalign|coh≥70={e['delta']:6.2f}  peak={e['peak']:6.2f}  n70={e['n70']}")
    print(f"\nplot  → {out_base}.png / .pdf")
    print(f"json  → {json_path}")


if __name__ == "__main__":
    sys.exit(main())
