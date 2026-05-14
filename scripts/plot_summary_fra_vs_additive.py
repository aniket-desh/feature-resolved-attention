#!/usr/bin/env python3
"""Phase 3 summary figure (paper-quality 1×2 frontier comparison).

Same SAE, same hookpoint, same prompts, single eval seed. Only the
intervention recipe differs.

Design choices (first-principles paper-style):

  - Wong colourblind-safe palette per method.
  - Trajectory = subtle line + filled circles at each α.
  - α value annotated outside each marker (small, grey).
  - Unsteered baseline = filled black star with white ring (one per panel).
  - coh = 70 floor drawn explicitly with a labelled vertical guide.
  - Δalign|coh≥70 surfaced as an in-axes range bracket (vertical bar
    with end-caps) on the right side of each method's curve, exactly at
    the alignment max/min of points with coh ≥ 70 — geometric, not
    textual. Per-method Δ printed in the legend so the reader can rank.
  - No floating stat box; legend carries the headline numbers.
  - Top/right spines off, axis tick density modest, light gridlines.
  - Sans-serif typography (Inter when available, Helvetica fallback).

Usage:
    python scripts/plot_summary_fra_vs_additive.py \
        --orig-per-seed-dir <dir>      # has aggregated_seed{seed}_medical.json
        --orig-additive    <gpt4o_aggregated_seed{seed}.json>
        --seed 42
        --out  <basename>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


def _bold_underline(s: str) -> str:
    """Return mathtext for bold-underlined `s` (mathtext doesn't ship `\\underline`,
    so we use a combining low-line on each character for the underline effect)."""
    underlined = "".join(c + "̲" for c in s)
    # Wrap in mathtext bold; mathtext renders unicode via `\mathrm{}` cleanly
    return r"$\mathbf{" + underlined.replace(" ", r"\ ") + r"}$"


# ─── Style ─────────────────────────────────────────────────────────────────
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
        "xtick.labelsize":     14,
        "ytick.labelsize":     14,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     13.5,
        "legend.title_fontsize": 14,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


# Wong palette assignments — colourblind-safe, distinct
PAL = {
    r"QK$\rightarrow$QK":    "#009E73",   # bluish green (the winner — easiest to distinguish)
    r"OV$\rightarrow$OV":    "#0072B2",   # blue
    r"QK$\rightarrow$OV":    "#D55E00",   # vermilion
    "additive": "#CC79A7",   # reddish purple
}


# ─── Data IO ───────────────────────────────────────────────────────────────
def load_method(path, method=None):
    d = json.loads(Path(path).read_text())
    if "aggregated" in d and isinstance(d["aggregated"], dict):
        d = d["aggregated"]
    if method:
        return sorted(d.get(method, []), key=lambda r: r["scale"])
    rows = []
    for m, r in d.items():
        if isinstance(r, list):
            rows.extend(r)
    return sorted(rows, key=lambda r: r["scale"])


def stats(rows, floor=COH_FLOOR):
    if not rows:
        return None
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    sc = np.array([r["scale"] for r in rows], dtype=float)
    mask = co >= floor
    if mask.any():
        al_m = al[mask]; co_m = co[mask]; sc_m = sc[mask]
        return dict(
            delta=float(al_m.max() - al_m.min()),
            min_align=float(al_m.min()),
            max_align=float(al_m.max()),
            n70=int(mask.sum()),
            x_at_max=float(co_m[al_m.argmax()]),
            x_at_min=float(co_m[al_m.argmin()]),
        )
    return dict(delta=float("nan"), min_align=float("nan"), max_align=float("nan"),
                n70=0, x_at_max=float("nan"), x_at_min=float("nan"))


# ─── Drawing primitives ────────────────────────────────────────────────────
def draw_method(ax, rows, color, label, *, marker="o", show_alpha=False,
                z_off=0):
    if not rows:
        return None
    sc = np.array([r["scale"] for r in rows])
    al = np.array([r["mean_alignment"] for r in rows])
    co = np.array([r["mean_coherence"] for r in rows])
    # subtle line trajectory
    ax.plot(co, al, color=color, lw=1.2, alpha=0.55, zorder=2 + z_off)
    # filled markers
    h = ax.scatter(co, al, marker=marker, s=120, facecolor=color,
                   edgecolor="white", linewidth=1.3, zorder=4 + z_off,
                   label=label)
    # α labels — only on the "winner" per panel
    if show_alpha:
        cx, cy = co.mean(), al.mean()
        for sval, x, y in zip(sc, co, al):
            dx, dy = x - cx, y - cy
            r = (dx * dx + dy * dy) ** 0.5 + 1e-6
            ox, oy = 9 * dx / r, 9 * dy / r
            ax.annotate(f"$\\alpha={sval:g}$",
                        xy=(x, y), xytext=(ox, oy),
                        textcoords="offset points",
                        fontsize=11, color="#444444",
                        ha="center", va="center", zorder=5 + z_off)
    return h


def draw_baseline_star(ax, x, y, label="unsteered"):
    ax.scatter([x], [y], marker="*", s=320, color="black",
               edgecolor="white", linewidth=1.3, zorder=7,
               label=label)


def draw_delta_bracket(ax, *, x, y_lo, y_hi, color, dx=2.5, lw=1.4):
    """Draw a vertical bracket on the RIGHT of x, spanning y_lo→y_hi.

    Marks the extent of Δalign within coh ≥ 70."""
    if not (y_hi > y_lo):
        return
    ax.plot([x + dx, x + dx], [y_lo, y_hi], color=color, lw=lw, zorder=3)
    ax.plot([x + dx - 0.6, x + dx + 0.6], [y_lo, y_lo], color=color, lw=lw, zorder=3)
    ax.plot([x + dx - 0.6, x + dx + 0.6], [y_hi, y_hi], color=color, lw=lw, zorder=3)


def decorate(ax, *, title, xlim, ylim):
    # coh = 70 floor
    ax.axvline(COH_FLOOR, color="#bbbbbb", lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.text(COH_FLOOR + 0.3, ylim[0] + 0.3, "coh = 70",
            color="#888888", fontsize=12, va="bottom", ha="left", zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 10))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 10))
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Alignment")
    ax.set_title(title, loc="center", pad=14, fontweight="600", color="#0a0a0a")
    ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-per-seed-dir", required=True)
    p.add_argument("--orig-additive", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--em-model", default="medical")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    setup_style()

    orig_path = Path(args.orig_per_seed_dir) / f"aggregated_seed{args.seed}_{args.em_model}.json"
    if not orig_path.exists():
        raise SystemExit(f"missing {orig_path}")
    additive_path = Path(args.orig_additive)
    if not additive_path.exists():
        raise SystemExit(f"missing {additive_path}")

    qkqk = load_method(orig_path, "qk_to_qk")
    qkov = load_method(orig_path, "qk_to_ov")
    ovov = load_method(orig_path, "ov_to_ov")
    baseline = load_method(orig_path, "baseline")
    additive = load_method(additive_path)

    # Compute joint axis range (round to nearest 10) accommodating both panels
    all_rows = qkqk + qkov + ovov + additive + (baseline or [])
    all_co = np.array([r["mean_coherence"] for r in all_rows])
    all_al = np.array([r["mean_alignment"] for r in all_rows])
    xlim = (int(np.floor(all_co.min() / 10) * 10),
            int(np.ceil(all_co.max() / 10) * 10))
    ylim = (int(np.floor(all_al.min() / 10) * 10),
            int(np.ceil(all_al.max() / 10) * 10))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10))

    # ── LEFT: three FRA recipes ──────────────────────────────────────────
    axL = axes[0]
    fra_methods = [
        (r"QK$\rightarrow$QK", qkqk, PAL[r"QK$\rightarrow$QK"], "o"),
        (r"OV$\rightarrow$OV", ovov, PAL[r"OV$\rightarrow$OV"], "s"),
        (r"QK$\rightarrow$OV", qkov, PAL[r"QK$\rightarrow$OV"], "D"),
    ]
    # Find the winner (largest Δ) so we only label its α values
    winner_label = None
    best_delta = -1
    for label, rows, _, _ in fra_methods:
        s = stats(rows)
        if s and s["n70"] and s["delta"] > best_delta:
            best_delta = s["delta"]; winner_label = label
    fra_stats = []
    for label, rows, color, marker in fra_methods:
        s = stats(rows)
        if s is None:
            continue
        if s["n70"] and s["delta"] == s["delta"]:
            v = f"{s['delta']:.2f}"
            d_str = r", Alignment $\Delta$ @ coh 70 = " + _bold_underline(v)
        else:
            d_str = r", Alignment $\Delta$ @ coh 70 = $\mathbf{\underline{NaN}}$"
        draw_method(axL, rows, color, label + d_str, marker=marker,
                    show_alpha=(label == winner_label))
        fra_stats.append((label, color, s))
    if baseline:
        b = baseline[0]
        draw_baseline_star(axL, b["mean_coherence"], b["mean_alignment"],
                           "baseline (no hook)")
    # Single Δ bracket — only on the winning recipe
    bracket_x = xlim[1] - 0.8
    for label, color, s in fra_stats:
        if label == winner_label and s["n70"] >= 1 and s["delta"] > 0:
            draw_delta_bracket(axL, x=bracket_x,
                               y_lo=s["min_align"], y_hi=s["max_align"],
                               color=color)
    decorate(axL, title="FRA decomposition", xlim=xlim, ylim=ylim)
    axL.legend(loc="upper left", frameon=True, fancybox=False,
               edgecolor="#222222", facecolor="white", framealpha=0.95)

    # ── RIGHT: conventional additive ────────────────────────────────────
    axR = axes[1]
    a_stats = stats(additive)
    if a_stats and a_stats["n70"] and a_stats["delta"] == a_stats["delta"]:
        v = f"{a_stats['delta']:.2f}"
        a_lab = r"additive, Alignment $\Delta$ @ coh 70 = " + _bold_underline(v)
    else:
        a_lab = r"additive, Alignment $\Delta$ @ coh 70 = $\mathbf{\underline{NaN}}$"
    draw_method(axR, additive, PAL["additive"], a_lab, marker="^",
                show_alpha=True)
    if additive:
        sc = np.array([r["scale"] for r in additive])
        i_one = int(np.argmin(np.abs(sc - 1.0)))
        draw_baseline_star(axR,
                           additive[i_one]["mean_coherence"],
                           additive[i_one]["mean_alignment"],
                           "α = 1.0 (no-op)")
    if a_stats and a_stats["n70"] >= 1 and a_stats["delta"] > 0:
        draw_delta_bracket(axR, x=bracket_x, y_lo=a_stats["min_align"],
                           y_hi=a_stats["max_align"], color=PAL["additive"])
    decorate(axR, title="Conventional steering", xlim=xlim, ylim=ylim)
    axR.legend(loc="upper left", frameon=True, fancybox=False,
               edgecolor="#222222", facecolor="white", framealpha=0.95)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
