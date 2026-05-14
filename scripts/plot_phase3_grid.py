#!/usr/bin/env python3
"""Phase 3: a single figure with one subplot per method.

Layout: 2 rows × 3 cols.
  cell (0,0): the original medical @ L24 ln1 — QK→OV / OV→OV / QK→QK overlaid
  cell (0,1): SAE-resid @ L24 hook_resid_pre  (all seeds)
  cell (0,2): SAE-resid @ L24 hook_resid_mid
  cell (1,0): SAE-resid @ L24 hook_resid_post
  cell (1,1): SAE-resid @ L25 ln1.hook_normalized
  cell (1,2): hidden

Each subplot:
  - alignment vs coherence (axes 0–100 each)
  - α-sweep trajectory(ies) coloured by seed (or by the original condition for cell 0,0)
  - α value annotated at each point
  - black star at α=0 (unsteered baseline)
  - reference lines at coh=70 (vertical) and align=50 (horizontal)
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


def _draw_axis(ax, *, title):
    ax.axvline(COH_FLOOR, color="grey", lw=0.7, ls=":", zorder=1)
    ax.axhline(50, color="grey", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("coherence")
    ax.set_ylabel("alignment")
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls=":", alpha=0.3)


def _draw_curve(ax, scales, al, co, color, label, *, lw=1.5, alpha=0.85,
                 annotate=True, marker_size=70):
    ax.plot(co, al, color=color, lw=lw, alpha=alpha, zorder=2)
    ax.scatter(co, al, c=color, s=marker_size, edgecolors="black",
               linewidths=0.5, zorder=3, label=label)
    if annotate:
        for sc, x, y in zip(scales, co, al):
            ax.annotate(f"{sc}", (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6, color="#333")
    # Black star at α=0
    i0 = int(np.argmin(np.abs(np.asarray(scales) - 0.0)))
    ax.scatter([co[i0]], [al[i0]], marker="*", s=260,
               color="black", edgecolors="white", linewidths=1.0, zorder=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-medical", required=True)
    p.add_argument("--sae", action="append", default=[], required=True)
    p.add_argument("--out", required=True, help="output base path (no extension)")
    p.add_argument("--title",
                   default="Phase 3 — alignment vs coherence frontier per method (medical EM)")
    args = p.parse_args()

    # ── Group SAE entries ────────────────────────────────────────────────
    by_label: dict[str, list] = defaultdict(list)
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        rows = sorted(load_method_rows(Path(path)), key=lambda r: r["scale"])
        if rows:
            by_label[label].append((seed, rows))

    # ── the original data (3 conditions) ────────────────────────────────────────
    orig_data = json.loads(Path(args.orig_medical).read_text())
    if "aggregated" in orig_data and isinstance(orig_data["aggregated"], dict):
        orig_data = orig_data["aggregated"]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    # Colour conventions
    orig_colours = {
        "qk_to_ov": ("#d73027", "QK→OV"),
        "ov_to_ov": ("#4575b4", "OV→OV"),
        "qk_to_qk": ("#1a9850", "QK→QK"),
    }
    seed_colours = {0: "#444", 42: "#4575b4", 123: "#1a9850",
                    456: "#d73027", 789: "#984ea3"}

    # ── Cell 0: the original combined ───────────────────────────────────────────
    ax = axes[0]
    for method, (color, lbl) in orig_colours.items():
        rows = sorted(orig_data.get(method, []), key=lambda r: r["scale"])
        if not rows:
            continue
        scales = np.array([r["scale"] for r in rows])
        al = np.array([r["mean_alignment"] for r in rows])
        co = np.array([r["mean_coherence"] for r in rows])
        _draw_curve(ax, scales, al, co, color, lbl)
    _draw_axis(ax, title="the original medical @ L24 ln1.hook_normalized\n(3-seed mean, 3 conditions)")
    ax.legend(loc="lower left", fontsize=8)

    # ── Cells 1..4: SAE-resid hookpoints ────────────────────────────────
    SAE_LAYOUT = [
        ("resid_pre_L24",     "blocks.24.hook_resid_pre"),
        ("resid_mid_L24",     "blocks.24.hook_resid_mid"),
        ("resid_post_L24",    "blocks.24.hook_resid_post"),
        ("ln1_normalised_L25","blocks.25.ln1.hook_normalized"),
    ]
    for cell_idx, (key, hookname) in enumerate(SAE_LAYOUT, start=1):
        ax = axes[cell_idx]
        entries = sorted(by_label.get(key, []), key=lambda x: x[0])
        for seed, rows in entries:
            scales = np.array([r["scale"] for r in rows])
            al = np.array([r["mean_alignment"] for r in rows])
            co = np.array([r["mean_coherence"] for r in rows])
            color = seed_colours.get(seed, "#444")
            _draw_curve(ax, scales, al, co, color, f"seed={seed}")
        _draw_axis(ax, title=f"SAE-resid @ {hookname}\n(medical EM, top-50 features)")
        ax.legend(loc="lower left", fontsize=8)

    # ── Cell 5: hidden ─────────────────────────────────────────────────
    axes[5].set_visible(False)

    # Single global star legend at the figure level
    from matplotlib.lines import Line2D
    star_handle = Line2D([], [], marker="*", color="black", linestyle="None",
                         markersize=14, markeredgecolor="white",
                         markeredgewidth=1.0, label="α=0 (unsteered)")
    fig.legend(handles=[star_handle], loc="upper right",
               bbox_to_anchor=(0.98, 0.97), frameon=True, fontsize=10)

    fig.suptitle(args.title, fontsize=13, y=1.0)
    fig.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
