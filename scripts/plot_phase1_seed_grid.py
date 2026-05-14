#!/usr/bin/env python3
"""Phase 1 cross-domain 3×5 seed × hookpoint grid (one figure per domain).

Rows = eval seeds {42, 123, 456}
Cols = 5 hookpoints
        0  L24 ln1 (published)            (3 FRA recipes overlaid + additive on published SAE)
        1  L24 resid_pre           (additive only)
        2  L24 resid_mid           (additive only)
        3  L24 resid_post          (additive only)
        4  L25 ln1                 (additive only)

Each cell = α-sweep alignment-vs-coherence trajectory for THAT (seed, hookpoint).
α annotated at each point; black star at the unsteered reference (the
published-SAE 'baseline' method for col 0, α=1.0 no-op for additive cols);
stats box with peak / baseline / min@coh70 / Δ@coh70.

Paper colour scheme (matches plot_phase1_fra_plus_additive.py):
  QK→QK = green   #009E73
  QK→OV = blue    #0072B2
  OV→OV = orange  #D55E00
  conventional steering (additive) = black #000000

Inputs: gpt4o_combined_*.json under --combined-root.

    python scripts/plot_phase1_seed_grid.py \
        --combined-root <dir>  --domain medical  --out <basename>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COH_FLOOR = 70.0
SEEDS = [42, 123, 456]

# Cols 1-4 are additive-only SAEs.
ADD_SAES = [
    ("L24_resid_pre",  "L24 resid_pre"),
    ("L24_resid_mid",  "L24 resid_mid"),
    ("L24_resid_post", "L24 resid_post"),
    ("L25_ln1",        "L25 ln1"),
]
# Col 0 = published SAE: FRA recipes + additive overlaid.
FRA_RECIPES = [
    ("qk_to_qk", r"QK$\rightarrow$QK", "#009E73"),
    ("qk_to_ov", r"QK$\rightarrow$OV", "#0072B2"),
    ("ov_to_ov", r"OV$\rightarrow$OV", "#D55E00"),
]
ADDITIVE_COLOR = "#000000"


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.0,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def load_combined(root: Path, domain: str):
    """Load every gpt4o_combined_*_{domain}.json under root.

    Returns dict keyed by 'sae_id' or 'FRA' with each value being the
    method→{by_alpha, summary} block.
    """
    out = {}
    keys = ["L24_ln1_published_FRA", "L24_ln1_published"] + [s for s, _ in ADD_SAES]
    for key in keys:
        p = root / f"gpt4o_combined_{key}_{domain}.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def _per_seed_curve(by_alpha, seed_idx):
    """Extract (scales, alignments, coherences) for one eval-seed slice."""
    scales = []
    al = []
    co = []
    for e in by_alpha:
        if seed_idx < len(e["per_seed_alignment"]):
            scales.append(e["scale"])
            al.append(e["per_seed_alignment"][seed_idx])
            co.append(e["per_seed_coherence"][seed_idx])
    order = np.argsort(scales)
    return (np.array(scales)[order], np.array(al)[order], np.array(co)[order])


def _summary_for_curve(scales, al, co, *, baseline_scale=1.0, floor=COH_FLOOR):
    out = {"peak": float(np.nanmax(al)) if al.size else float("nan"),
           "baseline": float("nan"),
           "min_above": float("nan"),
           "delta": float("nan"),
           "n_above": 0,
           "n_total": int(al.size)}
    if (scales == baseline_scale).any():
        out["baseline"] = float(al[scales == baseline_scale][0])
    mask = co >= floor
    if mask.any():
        out["min_above"] = float(al[mask].min())
        out["delta"] = float(al[mask].max() - al[mask].min())
        out["n_above"] = int(mask.sum())
    return out


def _draw_curve(ax, scales, al, co, color, *, alpha_for_line=0.85, marker="o", s=58, label=None):
    ax.plot(co, al, color=color, lw=1.4, alpha=alpha_for_line, zorder=2)
    ax.scatter(co, al, c=color, s=s, edgecolors="white", linewidths=0.8,
               zorder=3, marker=marker, label=label)
    for sc, x, y in zip(scales, co, al):
        ax.annotate(f"{sc}", (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=6, color="#444")


def _baseline_star(ax, co, al, label):
    ax.scatter([co], [al], marker="*", s=240,
               color="black", edgecolors="white", linewidths=0.9, zorder=6,
               label=label)


def _stat_box(ax, lines):
    if not lines:
        return
    line_h = 0.058
    pad_y = 0.025
    n = len(lines)
    box_top = 0.985
    box_bottom = box_top - line_h * n - pad_y
    box_left, box_right = 0.020, 0.640
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
            fontsize=8.5,
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
        ax.set_title(title, fontsize=9.5)
    ax.grid(True, ls=":", alpha=0.25)
    ax.tick_params(labelsize=8)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--domain", required=True, choices=["medical", "finance", "sports"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_style()

    combined = load_combined(Path(args.combined_root), args.domain)
    if not combined:
        raise SystemExit(f"No gpt4o_combined_*_{args.domain}.json found under {args.combined_root}")

    fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True, sharey=True)

    fra_block = combined.get("L24_ln1_published_FRA", {})
    orig_add  = combined.get("L24_ln1_published", {}).get("sae_resid", {}).get("by_alpha", [])

    for r, seed in enumerate(SEEDS):
        # ── Col 0: L24 ln1 (published) (FRA + additive overlaid) ───────────────
        ax = axes[r, 0]
        cond_stats = []
        for method, lbl, color in FRA_RECIPES:
            block = fra_block.get(method)
            if block is None:
                continue
            scales, al, co = _per_seed_curve(block["by_alpha"], r)
            if scales.size == 0:
                continue
            _draw_curve(ax, scales, al, co, color, label=f"FRA: {lbl}")
            s = _summary_for_curve(scales, al, co)
            cond_stats.append((f"FRA {lbl}", color, s))
        if orig_add:
            scales, al, co = _per_seed_curve(orig_add, r)
            if scales.size:
                _draw_curve(ax, scales, al, co, ADDITIVE_COLOR, label="Conventional",
                            marker="s", s=42)
                cond_stats.append(("Conventional", ADDITIVE_COLOR,
                                   _summary_for_curve(scales, al, co)))
        # baseline (no-hook) star — the published-SAE explicit baseline condition
        b = fra_block.get("baseline", {}).get("by_alpha", [])
        if b and r < len(b[0]["per_seed_alignment"]):
            be = b[0]
            _baseline_star(ax, be["per_seed_coherence"][r], be["per_seed_alignment"][r],
                           label="unsteered (no hook)")
        # Stats box: bold the winner Δ
        if cond_stats:
            valid = [(i, s["delta"]) for i, (_, _, s) in enumerate(cond_stats)
                     if s["n_above"] > 0 and s["delta"] == s["delta"]]
            winner_i = max(valid, key=lambda x: x[1])[0] if valid else -1
            lines = [("Δ@coh70 / peak (n above floor):", False)]
            for i, (lbl, col, s) in enumerate(cond_stats):
                if s["n_above"] > 0:
                    txt = f"  {lbl:14s}  Δ={s['delta']:5.2f}  pk={s['peak']:5.1f}  ({s['n_above']}/{s['n_total']})"
                else:
                    txt = f"  {lbl:14s}  Δ=NaN    pk={s['peak']:5.1f}  (0/{s['n_total']})"
                lines.append((txt, i == winner_i))
            _stat_box(ax, lines)
        title = f"L24 ln1 (published)\n(FRA + additive)\nseed={seed}" if r == 0 else f"seed={seed}"
        _decorate(ax, title=title, xlabel=(r == 2), ylabel=True,
                  legend_loc="lower right" if r == 0 else None)

        # ── Cols 1-4: surrounding-hookpoint additive ────────────────────
        for c, (sae_id, sae_lbl) in enumerate(ADD_SAES, start=1):
            ax = axes[r, c]
            block = combined.get(sae_id, {}).get("sae_resid", {}).get("by_alpha", [])
            if block:
                scales, al, co = _per_seed_curve(block, r)
                if scales.size:
                    _draw_curve(ax, scales, al, co, ADDITIVE_COLOR, label="Conventional",
                                marker="s", s=42)
                    # Black star at α=1.0 (math no-op of additive recipe)
                    if (scales == 1.0).any():
                        i_b = int(np.where(scales == 1.0)[0][0])
                        _baseline_star(ax, co[i_b], al[i_b], label=r"$\alpha=1.0$ (no-op)")
                    s = _summary_for_curve(scales, al, co)
                    if s["n_above"] > 0:
                        lines = [
                            (f"Δ@coh70 = {s['delta']:.2f}", True),
                            (f"peak | coh≥70 = {al[co >= COH_FLOOR].max():.1f}", False),
                            (f"peak overall = {s['peak']:.1f}", False),
                            (f"baseline (α=1) = {s['baseline']:.1f}", False),
                            (f"min | coh≥70 = {s['min_above']:.1f}", False),
                            (f"n@coh70 = {s['n_above']}/{s['n_total']}", False),
                        ]
                    else:
                        lines = [
                            ("Δ@coh70 = NaN", True),
                            (f"peak overall = {s['peak']:.1f}", False),
                            (f"baseline (α=1) = {s['baseline']:.1f}", False),
                            (f"n@coh70 = 0/{s['n_total']}", False),
                        ]
                    _stat_box(ax, lines)
            title = f"{sae_lbl}\n(additive)\nseed={seed}" if r == 0 else f"seed={seed}"
            _decorate(ax, title=title, xlabel=(r == 2), ylabel=False)

    fig.suptitle(f"Phase 1: alignment-vs-coherence trajectory per (seed × hookpoint), {args.domain}",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
