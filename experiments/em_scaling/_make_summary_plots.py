#!/usr/bin/env python3
"""
Paper-style headline plots for the EM-scaling sweep.

Reads `gpt4o_combined_FRA_<base>_<domain>.json` and
`gpt4o_combined_DoM_<base>_<domain>.json` per cell (produced by
``experiments.em_scaling.phase_judge combine``) and emits:

  1. ``frontier_panels.{png,pdf}``   2×3 alignment/coherence frontier
     per (base, domain) — one line per recipe, point per α.
  2. ``headline_bars.{png,pdf}``     per cell, best alignment @ coh≥70
     bars: FRA QK→QK vs DoM vs unsteered. Mirrors the paper's
     `phase1_fra_plus_additive_3domains.png` style.
  3. ``scaling.{png,pdf}``           Δ(alignment) @ coh≥70 vs model
     parameter count, lines for FRA QK→QK and DoM. Shows whether the
     QK→QK advantage grows / saturates with model scale.

Inputs are skipped if missing; the script is safe to call mid-chain.

Colour scheme matches ``scripts/plot_phase1_headline_per_domain.py`` so
the EM-scaling figures slot into the paper appendix cleanly:

  - FRA QK→QK     #009E73  (Wong bluish-green)
  - DoM (Turner)  #000000  (black, mirroring "conventional")
  - baseline      #BBBBBB  (light grey)
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

# Approximate parameter counts in B, for the scaling figure.
PARAM_COUNT_B = {
    "qwen-7b":  7.6,
    "llama-8b": 8.0,
    "qwen-14b": 14.7,
    "qwen-32b": 32.5,
}

COLOR = {
    "FRA":      "#009E73",
    "DoM":      "#000000",
    "baseline": "#BBBBBB",
}


def setup_style():
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":            14,
        "axes.titlesize":       16,
        "axes.labelsize":       14,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       1.2,
        "axes.edgecolor":       "#222222",
        "xtick.color":          "#222222",
        "ytick.color":          "#222222",
        "xtick.labelsize":      11,
        "ytick.labelsize":      12,
        "legend.frameon":       True,
        "legend.fontsize":      11,
        "figure.dpi":           110,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.1,
    })


def _combined_path(recipe: str, base: str, domain: str) -> Path:
    stage = "phase1_fra" if recipe == "FRA" else "phase2_dom"
    return Path(f"logs/em_scaling/{stage}/gpt4o_combined_{recipe}_{base}_{domain}.json")


def _load(recipe: str, base: str, domain: str):
    p = _combined_path(recipe, base, domain)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _winner(combined: dict, coh_floor: float = 70.0):
    if combined is None:
        return None
    best = None
    for method, body in combined.get("by_method", {}).items():
        if method == "baseline":
            continue
        for e in body["by_alpha"]:
            c = e.get("mean_coherence_across_seeds", e.get("mean_coherence"))
            if c is None or c < coh_floor:
                continue
            a = e.get("mean_alignment_across_seeds", e.get("mean_alignment"))
            if best is None or a > best["align"]:
                best = {"method": method, "scale": e["scale"], "align": a, "coh": c}
    return best


def _baseline(combined: dict):
    if combined is None:
        return None
    body = combined.get("by_method", {}).get("baseline")
    if not body or not body["by_alpha"]:
        return None
    e = body["by_alpha"][0]
    return {
        "align": e.get("mean_alignment_across_seeds", e.get("mean_alignment")),
        "coh":   e.get("mean_coherence_across_seeds", e.get("mean_coherence")),
    }


def _series(combined: dict, method: str | None = None):
    """Yield (scale, align, coh) for one method (the one non-baseline if None)."""
    if combined is None:
        return []
    by = combined.get("by_method", {})
    if method is None:
        method = next((m for m in by if m != "baseline"), None)
    if method is None or method not in by:
        return []
    out = []
    for e in by[method]["by_alpha"]:
        a = e.get("mean_alignment_across_seeds", e.get("mean_alignment"))
        c = e.get("mean_coherence_across_seeds", e.get("mean_coherence"))
        out.append((e["scale"], a, c))
    return out


def plot_frontier_panels(out_dir: Path, bases, domains):
    fig, axes = plt.subplots(len(bases), len(domains),
                             figsize=(4 * len(domains), 3.4 * len(bases)),
                             sharex=False, sharey=False)
    if len(bases) == 1 and len(domains) == 1:
        axes = np.array([[axes]])
    elif len(bases) == 1 or len(domains) == 1:
        axes = np.atleast_2d(axes)
    has_any = False
    for i, base in enumerate(bases):
        for j, domain in enumerate(domains):
            ax = axes[i, j]
            fra = _load("FRA", base, domain)
            dom = _load("DoM", base, domain)
            base_pt = _baseline(fra) or _baseline(dom)
            for recipe, combined in (("FRA", fra), ("DoM", dom)):
                pts = _series(combined)
                if not pts:
                    continue
                has_any = True
                pts.sort()
                xs = [p[2] for p in pts]   # coh
                ys = [p[1] for p in pts]   # align
                ax.plot(xs, ys, "-o", color=COLOR[recipe], lw=1.6, ms=5,
                        label=("FRA QK→QK" if recipe == "FRA" else "DoM"),
                        zorder=3)
            if base_pt is not None:
                ax.scatter([base_pt["coh"]], [base_pt["align"]],
                           s=90, color=COLOR["baseline"], edgecolor="#222",
                           zorder=4, label="unsteered")
            ax.axvline(70, ls=":", color="#999", lw=0.8, zorder=1)
            if i == 0:
                ax.set_title(domain.capitalize())
            if j == 0:
                ax.set_ylabel(f"{base}\nalignment")
            if i == len(bases) - 1:
                ax.set_xlabel("coherence")
            ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
    if has_any:
        axes[0, -1].legend(loc="best", fontsize=10)
    fig.suptitle("EM-scaling alignment / coherence frontier", fontsize=17, y=1.02)
    out = out_dir / "frontier_panels"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    return out, has_any


def plot_headline_bars(out_dir: Path, bases, domains, coh_floor: float = 70.0):
    cells = []
    for base in bases:
        for domain in domains:
            fra = _load("FRA", base, domain)
            dom = _load("DoM", base, domain)
            if fra is None and dom is None:
                continue
            cells.append({
                "base": base, "domain": domain,
                "fra": _winner(fra, coh_floor),
                "dom": _winner(dom, coh_floor),
                "baseline": _baseline(fra) or _baseline(dom),
            })
    if not cells:
        return None, False

    width = 0.36
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cells)), 5.8))

    # paper convention: Δ alignment @ coh≥70 = winner_alignment - baseline_alignment
    def delta(c, recipe):
        w = c[recipe]; b = c["baseline"]
        if w is None or b is None:
            return None
        return w["align"] - b["align"]

    fra_y = [delta(c, "fra") or 0.0 for c in cells]
    dom_y = [delta(c, "dom") or 0.0 for c in cells]

    b_fra = ax.bar(x - width/2, fra_y, width, color=COLOR["FRA"],
                   edgecolor="#222", linewidth=1.0, label="FRA QK→QK", zorder=3)
    b_dom = ax.bar(x + width/2, dom_y, width, color=COLOR["DoM"],
                   edgecolor="#222", linewidth=1.0, label="DoM (Turner)", zorder=3)
    for bar, m, d in list(zip(b_fra, fra_y, [delta(c, "fra") for c in cells])) + \
                       list(zip(b_dom, dom_y, [delta(c, "dom") for c in cells])):
        if d is None:
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                    "—", ha="center", va="bottom",
                    fontsize=11, color="#888", zorder=5)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, m + 0.6,
                    f"{m:+.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#0a0a0a", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['base']}\n{c['domain']}" for c in cells],
                       fontsize=11)
    ax.set_ylabel(f"Δ alignment @ coh ≥ {int(coh_floor)}  (winner − unsteered)")
    ax.set_title("EM-scaling headline: FRA QK→QK vs DoM (paper convention: Δ over unsteered)")
    ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    y_top = max([*fra_y, *dom_y, 10]) * 1.15
    ax.set_ylim(0, y_top)
    ax.legend(loc="best")

    out = out_dir / "headline_bars"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    return out, True


def plot_scaling(out_dir: Path, bases, domains, coh_floor: float = 70.0):
    """Δ alignment @ coh≥70 (winner − baseline) vs param count, line per recipe.
    Averages over domains per base when multiple are available."""
    points: dict[str, list[tuple[float, float, int]]] = {"FRA": [], "DoM": []}
    for base in bases:
        deltas_by_recipe = {"FRA": [], "DoM": []}
        for domain in domains:
            for recipe in ("FRA", "DoM"):
                combined = _load(recipe, base, domain)
                if combined is None:
                    continue
                w = _winner(combined, coh_floor)
                bsl = _baseline(combined)
                if w is None or bsl is None:
                    continue
                deltas_by_recipe[recipe].append(w["align"] - bsl["align"])
        for recipe, vals in deltas_by_recipe.items():
            if vals and base in PARAM_COUNT_B:
                points[recipe].append(
                    (PARAM_COUNT_B[base], float(np.mean(vals)), len(vals)))

    if not any(points.values()):
        return None, False

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for recipe in ("FRA", "DoM"):
        pts = sorted(points[recipe])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=COLOR[recipe], lw=2.0, ms=8,
                label=("FRA QK→QK" if recipe == "FRA" else "DoM"))
        for x_, y_ in zip(xs, ys):
            ax.text(x_, y_ + 0.6, f"{y_:.1f}",
                    ha="center", va="bottom", fontsize=10, color="#444")

    ax.set_xscale("log")
    ax.set_xlabel("model parameter count (B, log scale)")
    ax.set_ylabel(f"Δ alignment @ coh ≥ {int(coh_floor)}  (winner − unsteered)")
    ax.set_title("EM-scaling: steering advantage vs model size")
    ax.grid(True, which="both", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    out = out_dir / "scaling"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    return out, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="experiments/em_scaling/figures")
    p.add_argument("--bases", nargs="+", default=list(BASES))
    p.add_argument("--domains", nargs="+", default=list(DOMAINS))
    p.add_argument("--coh-floor", type=float, default=70.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    for name, fn in (
        ("frontier panels", lambda: plot_frontier_panels(out_dir, args.bases, args.domains)),
        ("headline bars",   lambda: plot_headline_bars(out_dir, args.bases, args.domains, args.coh_floor)),
        ("scaling",         lambda: plot_scaling(out_dir, args.bases, args.domains, args.coh_floor)),
    ):
        out, ok = fn()
        if ok:
            print(f"[plots] {name:<18s} → {out}.png / .pdf")
        else:
            print(f"[plots] {name:<18s} skipped (no combined judge data yet)")


if __name__ == "__main__":
    main()
