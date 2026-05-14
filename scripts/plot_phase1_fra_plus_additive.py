#!/usr/bin/env python3
"""1×2 cross-domain bar chart combining FRA recipes (on the published L24 ln1 SAE)
with additive recipe on each of 5 pre-trained SAEs.

8 bars per panel (one per (recipe, SAE)):

  FRA on published L24 ln1 SAE: qk_to_ov, ov_to_ov, qk_to_qk
  Additive:           L24 ln1 (published), L24 resid_pre, L24 resid_mid,
                       L24 resid_post, L25 ln1

Bar height = mean Δalign|coh≥70 across 3 eval seeds; error bar = sample std
(ddof=1). Bars labelled with Δ. Where the unsteered (no-hook) baseline
alignment is available, it's drawn as a horizontal dashed line per panel.

Inputs: gpt4o_combined_*.json files produced by phase1_judge_and_combine.py
(the new format with {by_alpha, summary} per method).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Paper-consistent colour scheme:
#   QK->QK : green   (#009E73 Wong bluish-green)
#   QK->OV : blue    (#0072B2 Wong blue)
#   OV->OV : orange  (#D55E00 Wong vermilion)
#   conventional steering : near-black (#1a1a1a)
FRA_COLUMNS = [
    ("FRA:qk_to_ov", r"FRA: QK$\rightarrow$OV", "#0072B2"),
    ("FRA:ov_to_ov", r"FRA: OV$\rightarrow$OV", "#D55E00"),
    ("FRA:qk_to_qk", r"FRA: QK$\rightarrow$QK", "#009E73"),
]
ADDITIVE_SAES = [
    ("L24_ln1_published",   "L24 ln1 (published)"),
    ("L24_resid_pre",  "L24 resid_pre"),
    ("L24_resid_mid",  "L24 resid_mid"),
    ("L24_resid_post", "L24 resid_post"),
    ("L25_ln1",        "L25 ln1"),
]
ADDITIVE_COLOR = "#000000"


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
        "xtick.labelsize":     11.5,
        "ytick.labelsize":     14,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      True,
        "legend.fontsize":     11.5,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def load_metrics(streams_root: Path):
    """Return {(col_key, em): summary_dict}."""
    out = {}
    for em in ("medical", "finance", "sports"):
        # FRA recipes from gpt4o_combined_L24_ln1_published_FRA_<em>.json
        fra_path = streams_root / f"gpt4o_combined_L24_ln1_published_FRA_{em}.json"
        if fra_path.exists():
            d = json.loads(fra_path.read_text())
            for method in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
                if method in d:
                    out[(f"FRA:{method}", em)] = d[method]["summary"]
            # Also pull the no-hook baseline for the horizontal dashed line
            if "baseline" in d:
                e = d["baseline"]["by_alpha"][0]
                out[("BASELINE_NOHOOK", em)] = {
                    "alignment": e["mean_alignment_across_seeds"],
                    "alignment_std": e["std_alignment_across_seeds"],
                    "coherence": e["mean_coherence_across_seeds"],
                }
        # Additive on each SAE
        for sae in ("L24_ln1_published", "L24_resid_pre", "L24_resid_mid",
                    "L24_resid_post", "L25_ln1"):
            p = streams_root / f"gpt4o_combined_{sae}_{em}.json"
            if p.exists():
                d = json.loads(p.read_text())
                if "sae_resid" in d:
                    out[(f"ADD:{sae}", em)] = d["sae_resid"]["summary"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_style()

    metrics = load_metrics(Path(args.combined_root))
    print("loaded keys:", sorted(metrics.keys()))

    domains = [d for d in ("medical", "finance", "sports")
               if any(em == d for _, em in metrics.keys())]
    fig, axs = plt.subplots(1, len(domains), figsize=(7.5 * len(domains), 6.8), sharey=True)
    if len(domains) == 1:
        axs = [axs]

    # joint y-lim across all FRA + best-additive bars
    all_max = []
    for em in domains:
        # FRA candidates
        for col_key, _, _ in FRA_COLUMNS:
            s = metrics.get((col_key, em))
            if s is not None:
                d = s["delta_coh_70"]
                if d["mean"] is not None:
                    all_max.append(d["mean"] + (d["std"] or 0))
        # Best additive candidate
        for sae, _ in ADDITIVE_SAES:
            s = metrics.get((f"ADD:{sae}", em))
            if s is not None:
                d = s["delta_coh_70"]
                if d["mean"] is not None:
                    all_max.append(d["mean"] + (d["std"] or 0))
    y_top = max(all_max + [10.0]) * 1.4
    y_top = max(int(np.ceil(y_top / 5) * 5), 25)

    for ax, em in zip(axs, domains):
        # Build the 4 bars: 3 FRA + 1 best-additive
        labels, means, stds, ns, colors = [], [], [], [], []
        for col_key, label, color in FRA_COLUMNS:
            s = metrics.get((col_key, em))
            if s is None:
                m, sd, n = 0.0, 0.0, 0
            else:
                d = s["delta_coh_70"]
                m  = d["mean"] if d["mean"] is not None else 0.0
                sd = d["std"]  if d["std"]  is not None else 0.0
                n  = d["n"]
            labels.append(label); means.append(m); stds.append(sd); ns.append(n); colors.append(color)
        # Pick best additive among the 5 SAEs (count only cells with n>0)
        add_candidates = []
        for sae, sae_lbl in ADDITIVE_SAES:
            s = metrics.get((f"ADD:{sae}", em))
            if s is None:
                continue
            d = s["delta_coh_70"]
            if d["n"] > 0 and d["mean"] is not None:
                add_candidates.append((sae_lbl, d["mean"], d["std"] or 0.0, d["n"]))
        if add_candidates:
            sae_lbl, m, sd, n = max(add_candidates, key=lambda c: c[1])
        else:
            sae_lbl, m, sd, n = "(none)", 0.0, 0.0, 0
        labels.append("Best conventional")
        means.append(m); stds.append(sd); ns.append(n); colors.append(ADDITIVE_COLOR)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds,
                      color=colors,
                      edgecolor="#222222", linewidth=0.8,
                      error_kw=dict(ecolor="#222222", capsize=5, capthick=1.2, lw=1.4),
                      zorder=3)
        for bar, m, sd, n in zip(bars, means, stds, ns):
            label_str = f"{m:.1f}" if n > 0 else "n=0"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (sd or 0) + 0.6,
                    label_str, ha="center", va="bottom",
                    fontsize=12, fontweight="600", color="#0a0a0a", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_title(em.capitalize(), loc="center", fontsize=18, fontweight="bold")
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, y_top)
        # vertical separator between FRA group and the best-additive bar
        n_fra = len(FRA_COLUMNS)
        sep_fra = n_fra - 0.5
        ax.axvline(sep_fra, color="#bbbbbb", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.text(sep_fra - 0.05, y_top * 0.95, "FRA decomposition",
                ha="right", va="top", fontsize=11.5, color="#555555", fontweight="600")
        ax.text(sep_fra + 0.05, y_top * 0.95, "Best conventional additive",
                ha="left", va="top", fontsize=11.5, color="#555555", fontweight="600")
        # baseline (no-hook) reference: top-left, broken across two lines
        b = metrics.get(("BASELINE_NOHOOK", em))
        if b is not None and b["alignment"] is not None:
            ax.text(0.02, 0.85,
                    f"Unsteered baseline\nalign = {b['alignment']:.1f} ± {b['alignment_std']:.2f}, coh = {b['coherence']:.1f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10.5, color="#444",
                    bbox=dict(facecolor="white", edgecolor="#222", boxstyle="round,pad=0.4"))

    axs[0].set_ylabel(r"Alignment $\Delta$ @ coh 70")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
