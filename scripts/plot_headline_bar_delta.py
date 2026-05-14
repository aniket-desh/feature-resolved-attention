#!/usr/bin/env python3
"""Headline bar chart: per-method Δalign|coh≥70 (mean ± std across eval seeds).

One bar per method. Bar height = mean of per-seed Δalign|coh≥70, error
bars = std across the 3 eval seeds (sample std, ddof=1). FRA decomposition
methods get their per-recipe colours (matching the frontier figure);
conventional additive methods all share one distinct colour.

Inputs:
  --orig-qualitative <path>   one the original `qualitative_*.json` containing all
                               4 conditions (qk_to_ov, ov_to_ov, qk_to_qk,
                               baseline)
  --additive label=path       per-method qualitative for additive runs;
                               pass once per method

Output:
  --out <basename>            <basename>.{png,pdf}

Uses the same paper aesthetic as
fra_proj/scripts/plot_summary_fra_vs_additive.py (Wong palette,
top-right spines off, sans-serif typography, big text).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


# Wong palette
PAL_FRA = {
    r"QK$\rightarrow$QK": "#009E73",  # green (winner)
    r"OV$\rightarrow$OV": "#0072B2",  # blue
    r"QK$\rightarrow$OV": "#D55E00",  # vermilion
}
COL_CONVENTIONAL = "#CC79A7"  # reddish purple — single shared colour


def setup_style():
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           15,
        "axes.titlesize":      18,
        "axes.labelsize":      16,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.2,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#222222",
        "ytick.color":         "#222222",
        "xtick.labelsize":     12.5,
        "ytick.labelsize":     14,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      True,
        "legend.fontsize":     12,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def per_seed_deltas(qualitative_path, *, condition_prefix=None,
                    floor=COH_FLOOR):
    """Compute Δalign|coh≥floor for each eval seed.

    If condition_prefix is given (e.g. 'qk_to_ov'), only entries whose
    `condition` starts with that prefix are used; otherwise all entries
    not in `baseline` are used.
    """
    data = json.loads(Path(qualitative_path).read_text())
    by_seed = defaultdict(lambda: defaultdict(lambda: {"al": [], "co": []}))
    for ex in data:
        if "gpt4o_alignment" not in ex: continue
        cond = ex.get("condition", "")
        if cond == "baseline": continue
        if condition_prefix is not None and not cond.startswith(condition_prefix):
            continue
        s = int(ex["seed"]); a = float(ex["scale"])
        by_seed[s][a]["al"].append(ex["gpt4o_alignment"])
        by_seed[s][a]["co"].append(ex["gpt4o_coherence"])

    deltas = []
    for s in sorted(by_seed):
        scs = sorted(by_seed[s])
        al = np.array([np.mean(by_seed[s][a]["al"]) for a in scs])
        co = np.array([np.mean(by_seed[s][a]["co"]) for a in scs])
        mask = co >= floor
        if mask.any():
            deltas.append(float(al[mask].max() - al[mask].min()))
    return deltas


def parse_additive(s):
    parts = s.split("=", 1)
    if len(parts) != 2:
        raise SystemExit(f"--additive '{s}' must be label=path")
    return parts[0], parts[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-qualitative", required=True)
    p.add_argument("--additive", action="append", default=[], required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    setup_style()

    # ── Compute per-seed deltas for each method ─────────────────────────
    methods = []  # list of (label, color, group, deltas)
    for cond, lab, col in [
        ("qk_to_qk", r"QK$\rightarrow$QK", PAL_FRA[r"QK$\rightarrow$QK"]),
        ("ov_to_ov", r"OV$\rightarrow$OV", PAL_FRA[r"OV$\rightarrow$OV"]),
        ("qk_to_ov", r"QK$\rightarrow$OV", PAL_FRA[r"QK$\rightarrow$OV"]),
    ]:
        ds = per_seed_deltas(args.orig_qualitative, condition_prefix=cond)
        methods.append((lab, col, "FRA", ds))
    for entry in args.additive:
        label, path = parse_additive(entry)
        ds = per_seed_deltas(path)
        methods.append((label, COL_CONVENTIONAL, "conventional", ds))

    # ── Sort: FRA group first (by mean desc), then conventional (by mean desc)
    fra = [m for m in methods if m[2] == "FRA"]
    conv = [m for m in methods if m[2] == "conventional"]
    fra.sort(key=lambda m: -np.mean(m[3]) if m[3] else 1e9)
    conv.sort(key=lambda m: -np.mean(m[3]) if m[3] else 1e9)
    ordered = fra + conv

    labels  = [m[0] for m in ordered]
    colors  = [m[1] for m in ordered]
    means   = [np.mean(m[3]) if m[3] else 0.0 for m in ordered]
    # sample std (ddof=1) where >=2 valid seeds; use 0 for < 2.
    stds    = [np.std(m[3], ddof=1) if len(m[3]) >= 2 else 0.0 for m in ordered]
    n_seeds = [len(m[3]) for m in ordered]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors,
                  edgecolor="#222222", linewidth=0.8,
                  error_kw=dict(ecolor="#222222", capsize=5, capthick=1.2, lw=1.4),
                  zorder=3)

    # Value annotation on top of each bar
    for bar, m, s, n in zip(bars, means, stds, n_seeds):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + s + 1.0,
                f"{m:.1f}",
                ha="center", va="bottom",
                fontsize=13, fontweight="600", color="#0a0a0a", zorder=5)

    # Visual group separator (vertical dashed line between FRA and conventional);
    # group labels added once after y-lim is set, below.
    if fra and conv:
        sep_x = len(fra) - 0.5
        ax.axvline(sep_x, color="#bbbbbb", lw=0.9, ls=(0, (3, 3)), zorder=1)

    # Decorations
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(r"Alignment $\Delta$ @ coh 70")
    ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    # Auto y-lim: round up to nearest 5
    y_top = max(m + s for m, s in zip(means, stds)) + 4
    y_top = int(np.ceil(y_top / 5) * 5)
    ax.set_ylim(0, y_top)

    # Custom legend (the FRA section already has colours per method;
    # conventional bars are all one colour — annotate that)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PAL_FRA[r"QK$\rightarrow$QK"], edgecolor="#222", label=r"QK$\rightarrow$QK"),
        Patch(facecolor=PAL_FRA[r"OV$\rightarrow$OV"], edgecolor="#222", label=r"OV$\rightarrow$OV"),
        Patch(facecolor=PAL_FRA[r"QK$\rightarrow$OV"], edgecolor="#222", label=r"QK$\rightarrow$OV"),
        Patch(facecolor=COL_CONVENTIONAL, edgecolor="#222",
              label="conventional additive feature steering"),
    ]
    ax.legend(handles=handles, loc="upper right",
              frameon=True, fancybox=False,
              edgecolor="#222222", facecolor="white", framealpha=0.95)

    # Re-draw group separator labels now that we have y_top
    if fra and conv:
        for i, txt in enumerate([("FRA decomposition", -0.05, "right"),
                                 ("Conventional steering", 0.05, "left")]):
            label, dx, ha = txt
            ax.text((len(fra) - 0.5) + dx,
                    y_top * 0.95, label,
                    ha=ha, va="top", fontsize=13, color="#555555",
                    fontweight="600")

    # Footer-ish caption removed per global style; if needed for context:
    # error bars = sample std (ddof=1) across 3 eval seeds
    ax.text(0.5, -0.22, r"error bars = sample std across 3 eval seeds",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#666666")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")
    print(f"\nMethods (sorted within group):")
    for lab, col, group, ds in ordered:
        ds_str = ", ".join(f"{d:.2f}" for d in ds)
        print(f"  [{group:13s}] {lab:25s}  mean={np.mean(ds):6.2f}  std={np.std(ds, ddof=1) if len(ds)>=2 else 0:5.2f}  "
              f"n={len(ds)}  per-seed=[{ds_str}]")


if __name__ == "__main__":
    sys.exit(main())
