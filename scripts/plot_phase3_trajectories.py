#!/usr/bin/env python3
"""Phase 3 alignment-vs-coherence trajectories.

For each method (the original QK→OV + 4 SAE-resid hookpoints), plot the curve of
(mean_coherence, mean_alignment) across α∈{0, 0.5, 1.0, 1.5, 2.0, 3.0}, with a
distinct line per seed. Marks the coh=70 floor and the α=0 baseline.

This makes the QK/OV-vs-SAE-resid story visible at a glance:
  - the published QK→OV curve sits at high coherence (>70) across the full α sweep,
    moving alignment up by ~Δalign|coh≥70.
  - Most SAE-resid curves collapse below coh=70 fast, even though they reach
    higher peak alignment (their curves go up and to the LEFT).

Usage:
    python scripts/plot_phase3_trajectories.py \
        --orig-medical <path> \
        --sae label=path[=seed] [--sae label=path[=seed] ...] \
        --out <basename>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COH_FLOOR = 70.0


def _seed_from_filename(p: str) -> int | None:
    m = re.search(r"seed[_-]?(\d+)", p)
    return int(m.group(1)) if m else None


def parse_sae_arg(s: str):
    if "=" in s:
        parts = s.split("=", 2)
    else:
        raise SystemExit(f"--sae arg '{s}' must be label=path[=seed]")
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], int(parts[2])


def load_method_rows(path: Path, prefer_method: str | None = None):
    data = json.loads(path.read_text())
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data = data["aggregated"]
    if prefer_method:
        return data.get(prefer_method, [])
    rows = []
    for m, r in data.items():
        if isinstance(r, list):
            rows.extend(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-medical", required=True)
    p.add_argument("--orig-method", default="qk_to_ov")
    p.add_argument("--sae", action="append", default=[], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Phase 3 — alignment vs coherence trajectories (medical EM)")
    args = p.parse_args()

    # Group SAE entries by label so seeds plot together
    by_label: dict[str, list[tuple[int, list]]] = defaultdict(list)
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        rows = load_method_rows(Path(path))
        if rows:
            by_label[label].append((seed, sorted(rows, key=lambda r: r["scale"])))

    # the original goes first
    orig_rows = sorted(load_method_rows(Path(args.orig_medical), prefer_method=args.orig_method),
                       key=lambda r: r["scale"])
    methods_in_order = [("FRA QK→OV (the original)", [(0, orig_rows)])] + \
                       [(lbl, by_label[lbl]) for lbl in
                        ["resid_pre_L24", "resid_mid_L24", "resid_post_L24", "ln1_normalised_L25"]
                        if lbl in by_label]

    # Layout: 2 rows × 3 cols (one cell per method, leaves 1 empty)
    n = len(methods_in_order)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    seed_colours = {0: "#d73027", 42: "#4575b4", 123: "#1a9850", 456: "#984ea3"}
    method_colours = {
        "FRA QK→OV (the original)": "#d73027",
        "resid_pre_L24": "#4575b4",
        "resid_mid_L24": "#1a9850",
        "resid_post_L24": "#984ea3",
        "ln1_normalised_L25": "#f0883e",
    }

    for ax, (label, entries) in zip(axes, methods_in_order):
        for seed, rows in entries:
            scales = np.array([r["scale"] for r in rows])
            al = np.array([r["mean_alignment"] for r in rows])
            co = np.array([r["mean_coherence"] for r in rows])
            color = method_colours.get(label, "k")
            alpha_line = 0.9 if seed == 0 else 0.7  # solid for the original
            ls = "-" if seed in (0, 42) else "--"
            ax.plot(co, al, ls=ls, color=color, lw=1.6, alpha=alpha_line,
                    label=f"seed={seed}" if label != "FRA QK→OV (the original)" else "3-seed mean")
            ax.scatter(co, al, c=color, edgecolors="black", linewidths=0.4,
                       s=70, zorder=4, alpha=alpha_line)
            for sc, x, y in zip(scales, co, al):
                ax.annotate(f"α={sc}", (x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color="#333")
        ax.axvline(COH_FLOOR, color="grey", lw=0.6, ls=":", zorder=1)
        ax.axhline(50, color="grey", lw=0.4, ls=":", zorder=1)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("coherence")
        ax.set_ylabel("alignment")
        ax.grid(True, ls=":", alpha=0.3)
        if entries:
            ax.legend(loc="lower left", fontsize=8)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(args.title, fontsize=13, y=1.0)
    fig.tight_layout()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
