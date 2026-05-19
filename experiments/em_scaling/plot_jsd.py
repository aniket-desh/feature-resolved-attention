#!/usr/bin/env python3
"""
JSD curves per em_scaling cell: FRA QK→QK vs DoM α-sweep.

Reads ``logs/em_scaling/phase3_jsd/jsd_<base>_<domain>.json`` and plots
JSD-vs-α per cell. Two outputs:

  - ``figures/jsd_panels.{png,pdf}``  one panel per cell, two curves
  - ``figures/jsd_headline.{png,pdf}`` 2×3 grid (base × domain), DoM line
    overlaid on FRA line; identifies α-budget where DoM matches FRA's JSD

Important α-axis caveat: FRA's α=0 means *multiply features by 0 =
ablate*, the most aggressive cell. DoM's α=0 is a no-op. So we plot
both on the same x-axis but annotate the FRA "identity" cell at α=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BASES = ("qwen-7b", "llama-8b", "qwen-14b", "qwen-32b")
DOMAINS = ("medical", "finance", "sports")

COLOR = {
    "FRA": "#009E73",
    "DoM": "#000000",
}


def setup_style():
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":            13,
        "axes.titlesize":       15,
        "axes.labelsize":       13,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       1.2,
        "axes.edgecolor":       "#222222",
        "xtick.color":          "#222222",
        "ytick.color":          "#222222",
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "legend.frameon":       True,
        "legend.fontsize":      10,
        "figure.dpi":           110,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.10,
    })


def _load(base, domain):
    p = Path(f"logs/em_scaling/phase3_jsd/jsd_{base}_{domain}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _series(jsd_dict: dict) -> tuple[list[float], list[float]]:
    if not jsd_dict:
        return [], []
    items = sorted([(float(a), v) for a, v in jsd_dict.items()])
    return [a for a, _ in items], [v for _, v in items]


def plot_panels(out_dir: Path, bases, domains):
    fig, axes = plt.subplots(len(bases), len(domains),
                             figsize=(4.2 * len(domains), 3.2 * len(bases)),
                             sharex=False, sharey=False)
    if len(bases) == 1 and len(domains) == 1:
        axes = np.array([[axes]])
    elif len(bases) == 1 or len(domains) == 1:
        axes = np.atleast_2d(axes)

    has_any = False
    for i, base in enumerate(bases):
        for j, domain in enumerate(domains):
            ax = axes[i, j]
            d = _load(base, domain)
            if d is None:
                ax.set_title(f"{base} / {domain}\n(no data)", color="#888")
                ax.set_axis_off()
                continue
            has_any = True
            for recipe, key in (("FRA QK→QK", "fra_qk_to_qk"), ("DoM", "dom")):
                xs, ys = _series(d.get(key, {}))
                if not xs:
                    continue
                col = COLOR["FRA"] if recipe.startswith("FRA") else COLOR["DoM"]
                ax.plot(xs, ys, "-o", color=col, lw=1.8, ms=5, label=recipe)
            # FRA "identity" marker at α=1
            ax.axvline(1.0, ls=":", color="#bbb", lw=0.8, zorder=0)
            ax.set_title(f"{base} / {domain}", fontsize=12)
            if j == 0:
                ax.set_ylabel("JSD (bits) vs clean")
            if i == len(bases) - 1:
                ax.set_xlabel(r"$\alpha$")
            ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
    # legend on first non-empty axis
    if has_any:
        for ax in axes.ravel():
            if ax.has_data():
                ax.legend(loc="best", fontsize=9)
                break
    fig.suptitle("EM-scaling JSD curves: FRA QK→QK vs DoM per cell",
                 fontsize=17, y=1.02)
    out = out_dir / "jsd_panels"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    return out, has_any


def plot_headline(out_dir: Path, bases, domains):
    """JSD at the operating point chosen by each recipe's alignment winner.

    The point is: how much distributional drift is each recipe paying to
    get its alignment win? Lower JSD at the same alignment = more surgical.
    """
    rows = []
    for base in bases:
        for domain in domains:
            d = _load(base, domain)
            if d is None:
                continue
            # Pick the highest-α cell for each recipe as the "max effort" probe
            for recipe, key in (("FRA", "fra_qk_to_qk"), ("DoM", "dom")):
                jsd = d.get(key, {})
                if not jsd:
                    continue
                last_alpha = max(float(a) for a in jsd)
                rows.append({
                    "base": base, "domain": domain,
                    "recipe": recipe, "alpha": last_alpha,
                    "jsd": jsd[str(last_alpha)],
                })
    if not rows:
        return None, False

    cells = sorted({(r["base"], r["domain"]) for r in rows})
    width = 0.36
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cells)), 5.4))
    for k, recipe in enumerate(["FRA", "DoM"]):
        ys = []
        for c in cells:
            r = [x_ for x_ in rows if x_["base"] == c[0] and x_["domain"] == c[1] and x_["recipe"] == recipe]
            ys.append(r[0]["jsd"] if r else 0.0)
        bars = ax.bar(x + (k - 0.5) * width, ys, width, color=COLOR[recipe],
                      edgecolor="#222", linewidth=1.0,
                      label=("FRA QK→QK (highest α)" if recipe == "FRA"
                             else "DoM (highest α)"), zorder=3)
        for bar, m in zip(bars, ys):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, m + 0.015,
                        f"{m:.2f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#0a0a0a", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{d}" for b, d in cells], fontsize=11)
    ax.set_ylabel("JSD (bits) at max-α operating point")
    ax.set_title("EM-scaling JSD at the strongest steering setting per recipe")
    ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    out = out_dir / "jsd_headline"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    return out, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="experiments/em_scaling/figures")
    p.add_argument("--bases", nargs="+", default=list(BASES))
    p.add_argument("--domains", nargs="+", default=list(DOMAINS))
    args = p.parse_args()
    setup_style()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("panels",   lambda: plot_panels(out_dir, args.bases, args.domains)),
        ("headline", lambda: plot_headline(out_dir, args.bases, args.domains)),
    ):
        out, ok = fn()
        if ok:
            print(f"[jsd-plot] {name:<10s} → {out}.png / .pdf")
        else:
            print(f"[jsd-plot] {name:<10s} skipped (no JSD data yet)")


if __name__ == "__main__":
    main()
