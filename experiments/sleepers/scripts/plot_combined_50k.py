"""1×2 combined 50k figure — JSD (left) + layman rollout-level (right).

Only the 50k SAE row of the per-seed re-attributed sweep is rendered.

Left panel (distribution-level, "JSD"):
  green = JSD(steered, clean)     ↓ better
  red   = JSD(steered, poisoned)  ↑ better

Right panel (rollout-level, "layman"):
  green = clean-match rate (fraction of 200 prompts whose 16-token steered
          rollout matches the clean rollout word-for-word)  ↑ better
  red   = ASR-16 (fraction of prompts whose steered rollout contains the
          sleeper phrase)  ↓ better

Style matches `phase1_fra_plus_*` figures from the EM branch: Inter /
Helvetica sans-serif, hidden top/right spines, light-grey y-gridlines,
rounded "Unsteered baseline" annotation top-left of each panel.

Linestyle / marker:
  solid + circle    single OV → OV
  dashed + triangle conventional resid-mid additive

Output: writes <out>.png and <out>.pdf.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def setup_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          15,
        "axes.titlesize":     18,
        "axes.labelsize":     16,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#222222",
        "axes.labelcolor":    "#1a1a1a",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    13,
        "ytick.labelsize":    14,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     True,
        "legend.fontsize":    11,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def reduce_mean(values):
    if isinstance(values, list):
        return float(statistics.mean(values))
    return float(values)


def reduce_mean_minmax(values):
    """Return (mean, min, max). lo=hi=mean if a single value."""
    if isinstance(values, list):
        if len(values) >= 1:
            return (float(statistics.mean(values)),
                    float(min(values)),
                    float(max(values)))
        return 0.0, 0.0, 0.0
    return float(values), float(values), float(values)


def _band(ax, xs, lo_seq, hi_seq, color, alpha=0.18):
    """Shaded band between per-α (lo, hi) pairs.

    For n=3 SAE seeds, lo/hi are min/max of the seed-wise values — directly
    showing the data range rather than assuming Gaussian-style ±σ uncertainty.
    """
    ax.fill_between(xs, lo_seq, hi_seq, color=color, alpha=alpha, linewidth=0, zorder=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="JSON from jsd_2x2_sweep_saeseed.py with full metrics "
                        "(must include jsd_clean, jsd_pois, n_exact_match_clean, asr).")
    p.add_argument("--output", type=Path, required=True,
                   help="Path *without* extension; .png and .pdf are written.")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    setup_style()
    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    OV_KEY   = "ov_single_50k"
    CONV_KEY = "conventional_50k"

    # sleeper-plot palette — red/green preserves clean-vs-sleeper semantics
    GREEN = "#1a8a3f"
    RED   = "#c0322a"

    methods = [
        (OV_KEY,   r"single OV$\rightarrow$OV", "-",  "o"),
        (CONV_KEY, "conventional",              "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ─────────────────── LEFT PANEL — JSD (distribution-level) ───────────────────
    for key, name, linestyle, marker in methods:
        per_alpha = cfg[key]["per_alpha"]
        jc_mm = [reduce_mean_minmax(per_alpha[str(a)]["jsd_clean"]) for a in alphas]
        jp_mm = [reduce_mean_minmax(per_alpha[str(a)]["jsd_pois"])  for a in alphas]
        jc    = [m for m, _, _ in jc_mm]
        jc_lo = [lo for _, lo, _ in jc_mm];  jc_hi = [hi for _, _, hi in jc_mm]
        jp    = [m for m, _, _ in jp_mm]
        jp_lo = [lo for _, lo, _ in jp_mm];  jp_hi = [hi for _, _, hi in jp_mm]
        _band(ax_l, alphas, jc_lo, jc_hi, GREEN)
        _band(ax_l, alphas, jp_lo, jp_hi, RED)
        ax_l.plot(alphas, jc, color=GREEN, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  JSD(steered, clean)", zorder=3)
        ax_l.plot(alphas, jp, color=RED, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  JSD(steered, poisoned)", zorder=3)

    ax_l.axhline(1.0, color="#888888", linestyle=":", lw=0.9, alpha=0.7)
    ax_l.text(alphas[-1], 1.0 - 0.015, "JSD upper bound (1 bit)",
               fontsize=10.5, color="#666", ha="right", va="top")
    ax_l.set_ylabel("Jensen-Shannon divergence (bits)")
    ax_l.set_xlabel(r"steering coefficient  $\alpha$")
    ax_l.set_ylim(-0.04, 1.10)
    ax_l.set_xticks(alphas)
    ax_l.set_xticklabels([f"{a:.2g}" for a in alphas])
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb")

    # ─────────────────── RIGHT PANEL — rollout (layman) ───────────────────
    for key, name, linestyle, marker in methods:
        per_alpha = cfg[key]["per_alpha"]
        mr_mm = [reduce_mean_minmax([n / args.n_prompts
                                       for n in per_alpha[str(a)]["n_exact_match_clean"]])
                  for a in alphas]
        ar_mm = [reduce_mean_minmax(per_alpha[str(a)]["asr"]) for a in alphas]
        mr    = [m for m, _, _ in mr_mm]
        mr_lo = [lo for _, lo, _ in mr_mm];  mr_hi = [hi for _, _, hi in mr_mm]
        ar    = [m for m, _, _ in ar_mm]
        ar_lo = [lo for _, lo, _ in ar_mm];  ar_hi = [hi for _, _, hi in ar_mm]
        _band(ax_r, alphas, mr_lo, mr_hi, GREEN)
        _band(ax_r, alphas, ar_lo, ar_hi, RED)
        ax_r.plot(alphas, mr, color=GREEN, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  clean-match rate", zorder=3)
        ax_r.plot(alphas, ar, color=RED, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  sleeper rate (ASR)", zorder=3)

    ax_r.set_ylabel("Sleeper fraction / Word-word matches")
    ax_r.set_xlabel(r"steering coefficient  $\alpha$")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_xticks(alphas)
    ax_r.set_xticklabels([f"{a:.2g}" for a in alphas])
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb")

    if args.title:
        fig.suptitle(args.title, fontsize=15, y=1.00)

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out)
    if base.endswith(".png") or base.endswith(".pdf"):
        base = base.rsplit(".", 1)[0]
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
