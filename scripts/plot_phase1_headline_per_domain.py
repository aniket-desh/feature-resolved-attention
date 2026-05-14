#!/usr/bin/env python3
"""Compact headline figure: one bar per domain showing the *best* Δalign|coh≥70
across all FRA + additive recipes, labelled with the winning method.

Inputs: gpt4o_combined_*.json under --combined-root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


SAES = ["L24_ln1_published", "L24_resid_pre", "L24_resid_mid", "L24_resid_post", "L25_ln1"]

# Paper colour scheme — must match plot_phase1_fra_plus_additive.py
COLOR_BY_RECIPE = {
    "qk_to_qk":     "#009E73",  # green  ← QK→QK
    "qk_to_ov":     "#0072B2",  # blue   ← QK→OV
    "ov_to_ov":     "#D55E00",  # orange ← OV→OV
    "conventional": "#000000",  # pure black  ← conventional additive
}


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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
        "xtick.labelsize":     14,
        "ytick.labelsize":     14,
        "legend.frameon":      True,
        "legend.fontsize":     12,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def best_for_domain(root: Path, em: str):
    """Return (method_label, mean, std, n, color_key) for the recipe with the
    largest Δalign|coh≥70 in this domain. color_key is one of the keys in
    COLOR_BY_RECIPE so the plot can colour the bar by category.
    """
    candidates = []
    # FRA recipes from gpt4o_combined_L24_ln1_published_FRA_<em>.json
    fra = root / f"gpt4o_combined_L24_ln1_published_FRA_{em}.json"
    if fra.exists():
        d = json.loads(fra.read_text())
        for method in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
            if method in d:
                s = d[method]["summary"]["delta_coh_70"]
                arrow = r"$\rightarrow$"
                lbl = (f"FRA: QK{arrow}OV" if method == "qk_to_ov"
                       else f"FRA: OV{arrow}OV" if method == "ov_to_ov"
                       else f"FRA: QK{arrow}QK")
                candidates.append((lbl, s["mean"] or 0.0, s["std"] or 0.0, s["n"], method))
    # Additive on each SAE
    for sae in SAES:
        p = root / f"gpt4o_combined_{sae}_{em}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if "sae_resid" in d:
                s = d["sae_resid"]["summary"]["delta_coh_70"]
                lbl_map = {
                    "L24_ln1_published":   "Add: L24 ln1 (published)",
                    "L24_resid_pre":  "Add: L24 resid_pre",
                    "L24_resid_mid":  "Add: L24 resid_mid",
                    "L24_resid_post": "Add: L24 resid_post",
                    "L25_ln1":        "Add: L25 ln1",
                }
                candidates.append((lbl_map[sae], s["mean"] or 0.0, s["std"] or 0.0, s["n"], "conventional"))
    valid = [c for c in candidates if c[3] > 0]
    if not valid:
        return ("(none above floor)", 0.0, 0.0, 0, "conventional")
    return max(valid, key=lambda c: c[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--domains", nargs="+", default=["medical", "finance", "sports"])
    args = p.parse_args()
    setup_style()

    root = Path(args.combined_root)
    bests = [(em, *best_for_domain(root, em)) for em in args.domains]

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    x = np.arange(len(bests))
    means         = [b[2] for b in bests]
    stds          = [b[3] for b in bests]
    method_labels = [b[1] for b in bests]
    color_keys    = [b[5] for b in bests]
    domain_labels = [em.capitalize() for em, *_ in bests]

    bar_colors = [COLOR_BY_RECIPE[k] for k in color_keys]
    bars = ax.bar(x, means, yerr=stds,
                  color=bar_colors, edgecolor="#222222", linewidth=1.0,
                  error_kw=dict(ecolor="#222222", capsize=6, capthick=1.4, lw=1.6),
                  zorder=3)
    for bar, m, s, mlabel in zip(bars, means, stds, method_labels):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (s or 0) + 0.6,
                f"{m:.1f}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#0a0a0a", zorder=5)
        # method name inside the bar (vertical, white)
        ax.text(bar.get_x() + bar.get_width()/2, h / 2,
                mlabel,
                ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold",
                rotation=90, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=15, fontweight="600")
    ax.set_ylabel(r"Best alignment $\Delta$ @ coh 70")
    ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    y_top = max(m + (s or 0) for m, s in zip(means, stds)) + 5
    y_top = int(np.ceil(y_top / 5) * 5)
    ax.set_ylim(0, y_top)

    fig.text(0.5, -0.02,
             "best recipe per domain across {3 FRA decomposition + 5 conventional additive} candidates;\n"
             "error bars = sample std across 3 eval seeds",
             ha="center", va="top", fontsize=11, color="#666666")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
