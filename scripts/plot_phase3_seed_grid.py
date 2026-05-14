#!/usr/bin/env python3
"""Phase 3: 3 (rows = seeds) × 5 (cols = method/hookpoint) grid figure.

Cols:
  0  the original medical @ L24 ln1.hook_normalized       (QK→OV / OV→OV / QK→QK overlaid)
  1  SAE-resid @ blocks.24.hook_resid_pre
  2  SAE-resid @ blocks.24.hook_resid_mid
  3  SAE-resid @ blocks.24.hook_resid_post
  4  SAE-resid @ blocks.25.ln1.hook_normalized

Rows: seed = 42, 123, 456 (3 seeds total).

Each cell:
  - α-sweep alignment-vs-coherence trajectory
  - α annotated at each point
  - Black star at α=0 (unsteered)
  - coh=70 floor + align=50 reference lines

Usage:
    python scripts/plot_phase3_seed_grid.py \
        --orig-per-seed-dir <dir>     # contains aggregated_seed{42,123,456}_medical.json
        --sae label=path[=seed] [...]
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


COH_FLOOR = 70.0
SEEDS_ORDER = [42, 123, 456]
SAE_COLS = [
    ("resid_pre_L24",     "blocks.24.hook_resid_pre"),
    ("resid_mid_L24",     "blocks.24.hook_resid_mid"),
    ("resid_post_L24",    "blocks.24.hook_resid_post"),
    ("ln1_normalised_L25","blocks.25.ln1.hook_normalized"),
]
ORIG_CONDITIONS = [
    ("qk_to_ov", "QK→OV", "#d73027"),
    ("ov_to_ov", "OV→OV", "#4575b4"),
    ("qk_to_qk", "QK→QK", "#1a9850"),
]
# Optional 6th col when --include-orig-additive — the published SAE under our
# additive (α-1)·f·W_dec recipe at her hookpoint
ORIG_ADDITIVE_HOOK = "blocks.24.ln1.hook_normalized"


def _seed_from_filename(p: str):
    m = re.search(r"seed[_-]?(\d+)", p)
    return int(m.group(1)) if m else None


def parse_sae_arg(s: str):
    parts = s.split("=", 2)
    if len(parts) < 2:
        raise SystemExit(f"--sae '{s}' must be label=path[=seed]")
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], int(parts[2])


def load_method_rows(path: Path, prefer_method: str | None = None):
    data = json.loads(path.read_text())
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data = data["aggregated"]
    if prefer_method and prefer_method in data:
        return data[prefer_method]
    rows = []
    for m, r in data.items():
        if isinstance(r, list):
            rows.extend(r)
    return rows


def _draw_curve(ax, scales, al, co, color, label):
    """Draw one α-sweep trajectory.  No baseline star — that's drawn separately."""
    ax.plot(co, al, color=color, lw=1.4, alpha=0.85, zorder=2)
    ax.scatter(co, al, c=color, s=60, edgecolors="black",
               linewidths=0.4, zorder=3, label=label)
    for sc, x, y in zip(scales, co, al):
        ax.annotate(f"{sc}", (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=5.5, color="#333")


def _baseline_star(ax, co, al, *, label="unsteered (no-op)"):
    ax.scatter([co], [al], marker="*", s=240,
               color="black", edgecolors="white", linewidths=0.9, zorder=6,
               label=label)


def _stats(rows, floor=COH_FLOOR):
    """Per-curve summary: Δalign|coh≥floor, peak align|coh≥floor, peak overall, n70."""
    rows = sorted(rows, key=lambda r: r["scale"])
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    mask = co >= floor
    if mask.any():
        delta = float(al[mask].max() - al[mask].min())
        peak_at_floor = float(al[mask].max())
        n70 = int(mask.sum())
    else:
        delta = float("nan")
        peak_at_floor = float("nan")
        n70 = 0
    return {
        "delta": delta,
        "peak_at_floor": peak_at_floor,
        "peak_overall": float(np.nanmax(al)) if al.size else float("nan"),
        "n70": n70,
        "n_total": len(rows),
    }


def _stat_box(ax, lines):
    """Render a stacked stats box.  `lines` is a list of (text, is_bold) tuples."""
    if not lines:
        return
    # First, draw a background rectangle covering the whole block.
    line_h = 0.058
    pad_y = 0.025
    n = len(lines)
    box_top = 0.985
    box_bottom = box_top - line_h * n - pad_y
    box_left, box_right = 0.020, 0.560
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch(
        (box_left, box_bottom),
        box_right - box_left, box_top - box_bottom,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor="white", edgecolor="#888", linewidth=0.7,
        alpha=0.95, zorder=10,
    )
    ax.add_patch(bg)
    for i, (text, is_bold) in enumerate(lines):
        ax.text(
            box_left + 0.015,
            box_top - pad_y * 0.5 - i * line_h,
            text,
            transform=ax.transAxes,
            fontsize=9,
            family="DejaVu Sans",
            fontweight="bold" if is_bold else "normal",
            color="#111" if is_bold else "#333",
            verticalalignment="top",
            horizontalalignment="left",
            zorder=11,
        )


def _decorate(ax, *, title=None, xlabel=False, ylabel=False, legend_loc=None):
    ax.axvline(COH_FLOOR, color="grey", lw=0.6, ls=":", zorder=1)
    ax.axhline(50, color="grey", lw=0.4, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    if xlabel:
        ax.set_xlabel("coherence", fontsize=9)
    if ylabel:
        ax.set_ylabel("alignment", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, ls=":", alpha=0.25)
    ax.tick_params(labelsize=8)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-per-seed-dir", required=True,
                   help="dir with aggregated_seed{42,123,456}_medical.json")
    p.add_argument("--sae", action="append", default=[], required=True)
    p.add_argument("--orig-sae-additive", action="append", default=[],
                   help="<seed>=<gpt4o_aggregated_*.json> from running the published L24 ln1 SAE "
                        "through our additive (α-1)·f·W_dec recipe (sanity check). "
                        "Adds a 6th column.")
    p.add_argument("--orig-mode", default="combined",
                   choices=["combined", "qk_to_qk_only"],
                   help="combined = all 3 the original conditions overlaid (default); "
                        "qk_to_qk_only = show just QK→QK in col 0.")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    # Parse the original-additive entries (one per eval seed)
    orig_additive: dict[int, list] = {}
    for entry in args.orig_sae_additive:
        seed_str, path = entry.split("=", 1)
        seed = int(seed_str)
        rows = sorted(load_method_rows(Path(path)), key=lambda r: r["scale"])
        if rows:
            orig_additive[seed] = rows

    # ── Load the original per-seed ──────────────────────────────────────────────
    orig = {}
    for seed in SEEDS_ORDER:
        f = Path(args.orig_per_seed_dir) / f"aggregated_seed{seed}_medical.json"
        if f.exists():
            orig[seed] = json.loads(f.read_text())

    # ── Group SAE entries by (label, seed) ──────────────────────────────
    by_label_seed: dict[tuple[str, int], list] = {}
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        rows = sorted(load_method_rows(Path(path)), key=lambda r: r["scale"])
        if rows:
            by_label_seed[(label, seed)] = rows

    # ── Plot ────────────────────────────────────────────────────────────
    n_cols = 5 + (1 if orig_additive else 0)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 11),
                             sharex=True, sharey=True)

    show_only_qkqk = (args.orig_mode == "qk_to_qk_only")
    orig_conds_to_plot = (
        [("qk_to_qk", "QK→QK", "#1a9850")] if show_only_qkqk else ORIG_CONDITIONS
    )

    for r, seed in enumerate(SEEDS_ORDER):
        # Col 0: the original
        ax = axes[r, 0]
        orig_data = orig.get(seed, {})
        # Compute stats per condition first, find the winner
        cond_stats = []
        for method, label, color in orig_conds_to_plot:
            rows = orig_data.get(method, [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda rr: rr["scale"])
            scales = np.array([rr["scale"] for rr in rows])
            al = np.array([rr["mean_alignment"] for rr in rows])
            co = np.array([rr["mean_coherence"] for rr in rows])
            _draw_curve(ax, scales, al, co, color, label)
            cond_stats.append((label, _stats(rows)))
        # Single baseline star: the published 'baseline' method (no hook applied)
        baseline = orig_data.get("baseline", [])
        if baseline:
            b = baseline[0]
            _baseline_star(ax, b["mean_coherence"], b["mean_alignment"],
                           label="baseline (no hook)")
        if cond_stats:
            valid_deltas = [(i, s["delta"]) for i, (_, s) in enumerate(cond_stats)
                            if s["n70"] > 0 and s["delta"] == s["delta"]]
            winner_idx = max(valid_deltas, key=lambda x: x[1])[0] if valid_deltas else -1
            orig_lines = [("alignment delta @ coh ≥ 70:", False)]
            for i, (label, s) in enumerate(cond_stats):
                if s["n70"] > 0:
                    txt = f"  {label:6s}: Δ={s['delta']:5.2f}  peak={s['peak_at_floor']:5.2f}  ({s['n70']}/{s['n_total']})"
                else:
                    txt = f"  {label:6s}: Δ=NaN    peak=---     (0/{s['n_total']})"
                orig_lines.append((txt, i == winner_idx))
            _stat_box(ax, orig_lines)

        orig_title = (
            "the original medical QK→QK @ L24 ln1" if show_only_qkqk
            else "the original medical @ L24 ln1\n(QK→OV / OV→OV / QK→QK)"
        )
        title = f"{orig_title}\neval seed={seed}" if r == 0 else f"eval seed={seed}"
        _decorate(ax, title=title, xlabel=(r == 2), ylabel=True,
                  legend_loc="lower right" if r == 0 else None)

        # Cols 1..N: SAE-resid hookpoints, then optional the original-additive
        sae_columns = list(SAE_COLS)
        if orig_additive:
            sae_columns = sae_columns + [
                ("orig_additive", ORIG_ADDITIVE_HOOK + "  (the original SAE)")
            ]

        for c, (key, hookname) in enumerate(sae_columns, start=1):
            ax = axes[r, c]
            if key == "orig_additive":
                rows = orig_additive.get(seed, [])
            else:
                rows = by_label_seed.get((key, seed), [])
            if rows:
                scales = np.array([rr["scale"] for rr in rows])
                al = np.array([rr["mean_alignment"] for rr in rows])
                co = np.array([rr["mean_coherence"] for rr in rows])
                _draw_curve(ax, scales, al, co, "#222", f"eval seed={seed}")
                # Black star at α=1.0 (no-op for our steering rule (α-1)·f·W_dec)
                i_baseline = int(np.argmin(np.abs(scales - 1.0)))
                _baseline_star(ax, co[i_baseline], al[i_baseline],
                               label="α=1.0 (no-op)")
                s = _stats(rows)
                if s["n70"] > 0:
                    lines = [
                        (f"alignment delta @ coh ≥ 70 = {s['delta']:.2f}", True),
                        (f"peak alignment | coh ≥ 70 = {s['peak_at_floor']:.2f}", False),
                        (f"peak alignment overall    = {s['peak_overall']:.2f}", False),
                        (f"n @ coh ≥ 70 = {s['n70']}/{s['n_total']}", False),
                    ]
                else:
                    lines = [
                        ("alignment delta @ coh ≥ 70 = NaN", True),
                        ("peak alignment | coh ≥ 70 = ---", False),
                        (f"peak alignment overall    = {s['peak_overall']:.2f}", False),
                        (f"n @ coh ≥ 70 = 0/{s['n_total']}", False),
                    ]
                _stat_box(ax, lines)
            if key == "orig_additive":
                title = f"the original SAE + additive\n{hookname}" if r == 0 else None
            else:
                title = f"SAE-resid\n{hookname}" if r == 0 else None
            _decorate(ax, title=title, xlabel=(r == 2), ylabel=False)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=0.995)
    fig.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
