#!/usr/bin/env python3
"""1×2 cross-domain bar chart: Δalign|coh≥70 across 5 pre-trained SAEs for
finance + sports.

Each bar = mean over 3 eval seeds of per-seed Δalign|coh≥70 (computed on
the per-seed α-curve), error bar = sample std (ddof=1) across the 3 eval
seeds. Bars labelled with mean.

Inputs: gpt4o_combined_<sae_id>_<em_model>.json from
fra_proj/phase1_judge_and_combine.py.

Style: Wong colourblind-safe palette, Inter sans-serif, joint y-axis,
top + right spines off — matches `feedback_paper_figure_style.md` for FRA
frontier figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


# Wong palette + an extra colour for L25_ln1
SAE_COLOR = {
    "L24_ln1_published":   "#009E73",  # bluish green — hero (the published-SAE SAE)
    "L24_resid_pre":  "#0072B2",  # blue
    "L24_resid_mid":  "#D55E00",  # vermilion
    "L24_resid_post": "#CC79A7",  # reddish purple
    "L25_ln1":        "#E69F00",  # orange
}
SAE_LABEL = {
    "L24_ln1_published":   "L24 ln1 (published)",
    "L24_resid_pre":  "L24 resid_pre",
    "L24_resid_mid":  "L24 resid_mid",
    "L24_resid_post": "L24 resid_post",
    "L25_ln1":        "L25 ln1",
}
SAE_ORDER = ["L24_ln1_published", "L24_resid_pre", "L24_resid_mid",
             "L24_resid_post", "L25_ln1"]


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


def per_seed_deltas(combined_entries, floor=COH_FLOOR):
    """For one (sae_id, em_model) combined json, compute per-seed Δalign|coh≥floor.

    Returns list[float] of length n_seeds (skipping seeds with no α above floor).
    """
    n_seeds = combined_entries[0]["n_seeds"]
    per_seed = []
    for s in range(n_seeds):
        al, co = [], []
        for e in combined_entries:
            if s < len(e["per_seed_alignment"]):
                al.append(e["per_seed_alignment"][s])
                co.append(e["per_seed_coherence"][s])
        al = np.array(al); co = np.array(co)
        mask = co >= floor
        if mask.any():
            per_seed.append(al[mask].max() - al[mask].min())
    return per_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True,
                   help="dir containing gpt4o_combined_*.json")
    p.add_argument("--out", required=True, help="output basename (.png + .pdf)")
    p.add_argument("--method", default="sae_resid",
                   help="method key inside the combined files")
    args = p.parse_args()
    setup_style()

    root = Path(args.combined_root)
    rows = {}  # (sae_id, em_model) → list[float] of per-seed Δ
    for fn in root.glob("gpt4o_combined_*.json"):
        stem = fn.stem.replace("gpt4o_combined_", "")
        # Match against known SAE order so we strip the trailing _<em> safely
        sae_id, em = None, None
        for sid in SAE_ORDER:
            if stem.startswith(sid + "_"):
                sae_id = sid
                em = stem[len(sid) + 1:]
                break
        if sae_id is None:
            print(f"  [skip unknown] {fn.name}")
            continue
        data = json.loads(fn.read_text())
        if args.method not in data:
            print(f"  [skip missing method] {fn.name}")
            continue
        rows[(sae_id, em)] = per_seed_deltas(data[args.method])
    print(f"loaded {len(rows)} (sae, em) entries")

    domains = sorted({em for _, em in rows.keys()})
    print(f"domains: {domains}")

    fig, axs = plt.subplots(1, len(domains), figsize=(6.5 * len(domains), 6.4),
                             sharey=True)
    if len(domains) == 1:
        axs = [axs]

    # joint y-lim across panels
    all_means = []
    for v in rows.values():
        if v: all_means.append(float(np.mean(v)))
    y_top = max(all_means + [1.0]) * 1.6
    y_top = max(int(np.ceil(y_top / 5) * 5), 20)

    for ax, em in zip(axs, domains):
        x = np.arange(len(SAE_ORDER))
        means, stds, ns = [], [], []
        for sid in SAE_ORDER:
            d = rows.get((sid, em), [])
            if d:
                means.append(float(np.mean(d)))
                stds.append(float(np.std(d, ddof=1)) if len(d) >= 2 else 0.0)
                ns.append(len(d))
            else:
                means.append(0.0); stds.append(0.0); ns.append(0)
        bars = ax.bar(x, means, yerr=stds,
                      color=[SAE_COLOR[s] for s in SAE_ORDER],
                      edgecolor="#222222", linewidth=0.8,
                      error_kw=dict(ecolor="#222222", capsize=5, capthick=1.2, lw=1.4),
                      zorder=3)
        for bar, m, s, n in zip(bars, means, stds, ns):
            h = bar.get_height()
            label = f"{m:.1f}" if n > 0 else "n=0"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + s + 0.5,
                    label,
                    ha="center", va="bottom",
                    fontsize=12, fontweight="600", color="#0a0a0a", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([SAE_LABEL[s] for s in SAE_ORDER], rotation=22, ha="right")
        ax.set_title(em.capitalize(), loc="center", fontsize=18, fontweight="bold")
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, y_top)

    axs[0].set_ylabel(r"Alignment $\Delta$ @ coh 70")

    fig.text(0.5, -0.02,
             r"error bars = sample std across 3 eval seeds; bar height = mean per-seed $\Delta$alignment over $\alpha$ where coh $\geq$ 70",
             ha="center", va="top", fontsize=11, color="#666666")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
