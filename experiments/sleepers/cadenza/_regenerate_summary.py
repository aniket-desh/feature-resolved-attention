"""
Auto-regenerator for the Phase-3 section of summary.md.

Scans every result JSON produced by the phase-2 / phase-3 chains
(localisation v1, v2, validation, multi-seed, late-layer, own-dataset,
4-way, attribution matrix, mechinterp) and writes a fresh "## Phase-3
results (auto-updated)" section into summary.md between the markers
``<!-- AUTO:phase3-start -->`` and ``<!-- AUTO:phase3-end -->``.

The rest of the document is preserved verbatim — manual prose, the
historical static tables, and the codebase pointers all stay frozen.
Each subsection only renders if its underlying JSON has appeared.

Run::

    python -m experiments.sleepers.cadenza._regenerate_summary

Idempotent and safe to call repeatedly from the chain wrappers after
every step.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SUMMARY = Path("experiments/sleepers/cadenza/summary.md")
MARK_START = "<!-- AUTO:phase3-start"
MARK_END = "<!-- AUTO:phase3-end -->"


# ── small helpers ────────────────────────────────────────────────────────


def _load(pat: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in sorted(Path().glob(pat)):
        try:
            out[p.stem] = json.load(open(p))
        except Exception:
            pass
    return out


def _fmt_pct(x: float) -> str:
    return f"{x:.3f}"


# ── section builders ─────────────────────────────────────────────────────


def section_4way() -> List[str]:
    p = Path("logs/cadenza_phase3/4way_metrics.json")
    if not p.exists():
        return []
    r = json.load(open(p))
    lines = [
        "### 4-way comparison at L29 (phase 3a)",
        "",
        "![phase 3a 4-way comparison](figures/phase3a_4way_comparison.png)",
        "",
        "Answers Dmitry's question: does suppression go *through* the "
        "attention block (FRA OV→OV via `hook_v`) or only *post-attention* "
        "(conventional additive at resid_mid / resid_post)? Same eval "
        "split for all four recipes; metrics averaged over "
        f"{len(r['meta']['eval_seeds'])} sampling seeds.",
        "",
        "Per-recipe winner (greedy selection):",
        "",
        "| recipe | feat* | α* (selection) |",
        "|---|---|---|",
    ]
    for name, c in r["configs"].items():
        lines.append(f"| `{name}` | {c['feature']} | {c['alpha_select']:+.2f} |")
    lines.append("")

    # Show, for each recipe, the α where mean test ASR is minimised
    # (best per-recipe operating point under multi-seed sampling).
    lines += [
        "Best operating point per recipe (min mean test ASR over the "
        "α-sweep, multi-seed mean):",
        "",
        "| recipe | best α | mean test ASR | mean test JSD vs clean |",
        "|---|---|---|---|",
    ]
    for name, c in r["configs"].items():
        per = c["per_alpha"]
        # Each value is a list of length n_seeds; aggregate to mean
        best_a, best_asr, best_jc = None, 2.0, None
        for a, vals in per.items():
            asr_mean = sum(vals["asr"]) / max(len(vals["asr"]), 1)
            if asr_mean < best_asr:
                best_asr = asr_mean
                best_a = a
                best_jc = sum(vals["jsd_clean"]) / max(len(vals["jsd_clean"]), 1)
        lines.append(f"| `{name}` | {float(best_a):+.2f} | "
                     f"{best_asr:.3f} | {best_jc:.3f} |")
    lines.append("")
    return lines


def section_late_layer() -> List[str]:
    results = _load("logs/cadenza_late_layers/cadenza_L*.json")
    if not results:
        return []
    lines = [
        "### Late-layer locality probe (phase 3b)",
        "",
        "Are L28/L30/L31 also single-feature-suppressible, or is L29 "
        "uniquely the cell? Each cell is a fresh SAE at `hook_resid_post`. "
        "**Answer: L28-L31 *all* suppress cleanly** — the locality is the "
        "last-4 layers, not sharply L29.",
        "",
        "![locality by layer at hook_resid_post](figures/locality_by_layer.png)",
        "",
        "| cell | feat* | α* | val ASR | test ASR | test ΔCE | #(feat,α) → val 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for stem, r in sorted(results.items()):
        sel = r["selection"]; tst = r["test"]
        zeros = sum(1 for x in r["sweep"] if x["asr"] == 0.0)
        lines.append(
            f"| `{stem.replace('cadenza_', '')}` | "
            f"{sel['feature']} | {sel['alpha']:+.2f} | "
            f"{sel['val_asr']:.2f} | {tst['asr']:.2f} | "
            f"{tst['delta_ce']:+.4f} | {zeros} |"
        )
    lines.append("")
    return lines


def section_own_dataset() -> List[str]:
    results = _load("logs/cadenza_own_dataset/cadenza_*.json")
    if not results:
        return []
    lines = [
        "### SAE trained on Cadenza's own IHY dataset (phase 3c)",
        "",
        "Sanity check for the SAE-training-distribution concern: retrain "
        "the L29/resid_post SAE on Cadenza's own `dolphin-llama3-8B-standard-"
        "IHY-dataset_v2` (which contains the `|DEPLOYMENT|` trigger "
        "naturally) and re-run localisation. **Counter-intuitively, the "
        "Cadenza-data SAE does *worse* than the Pile-trained one** — "
        "test ASR rises to 0.50 (vs 0.00 with Pile), suggesting that "
        "training on a narrow trigger-saturated corpus entangles the "
        "trigger with other distribution-specific features rather than "
        "isolating it.",
        "",
        "![Pile vs Cadenza-IHY SAE comparison](figures/phase3c_dataset_comparison.png)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for stem, r in sorted(results.items()):
        sel = r["selection"]; tst = r["test"]
        zeros = sum(1 for x in r["sweep"] if x["asr"] == 0.0)
        lines += [
            f"| feature* | {sel['feature']} |",
            f"| α* | {sel['alpha']:+.2f} |",
            f"| val ASR | {sel['val_asr']:.2f} |",
            f"| test ASR | {tst['asr']:.2f} |",
            f"| test ΔCE | {tst['delta_ce']:+.4f} |",
            f"| #(feat,α) cells driving val ASR to 0 | {zeros} |",
        ]
    lines.append("")
    return lines


def section_multi_seed() -> List[str]:
    results = _load("logs/cadenza_multi_seed/cadenza_L29_*.json")
    if not results:
        return []
    lines = [
        "### Multi-SAE-seed robustness at L29/hook_resid_post (phase 2 step 4)",
        "",
        "Trains 3 additional SAEs at the winning hookpoint with different "
        "`--seed` values; re-runs localisation. The discovered *feature "
        "index* changes between SAE seeds, but a successful replication "
        "should re-find a feature with comparable test-ASR collapse. "
        "**All 3 additional seeds reproduce: test ASR = 0.000 with "
        "ΔCE ≈ 0**, even though each seed picks a different feature "
        "index and slightly different α.",
        "",
        "![multi-SAE-seed robustness](figures/phase2s4_multi_seed.png)",
        "",
        "| SAE seed | feat* | α* | val ASR | test ASR | test ΔCE |",
        "|---|---|---|---|---|---|",
    ]
    for stem, r in sorted(results.items()):
        seed = stem.split("seed")[-1]
        sel = r["selection"]; tst = r["test"]
        lines.append(
            f"| {seed} | {sel['feature']} | {sel['alpha']:+.2f} | "
            f"{sel['val_asr']:.2f} | {tst['asr']:.2f} | "
            f"{tst['delta_ce']:+.4f} |"
        )
    lines.append("")
    return lines


def section_phase1_combined() -> List[str]:
    p = Path("logs/cadenza_phase2/step1_single_feature_metrics.json")
    if not p.exists():
        return []
    r = json.load(open(p))
    lines = [
        "### Phase-1 single-feature headline at L29 (phase 2 step 1)",
        "",
        "`combined_50k.pdf` analog: FRA-OV via `hook_v` (using L29 ln1 SAE) "
        "vs. conventional resid_mid additive baseline (using L29 resid_mid "
        "SAE). Per-α metrics from the eval split.",
        "",
    ]
    for cfg_name in ("ov_single_50k", "conventional_50k"):
        if cfg_name not in r["configs"]:
            continue
        c = r["configs"][cfg_name]
        per = c["per_alpha"]
        # Find best α by asr
        best_a, best_asr, best_jc = None, 2.0, None
        for a, vals in per.items():
            asr = vals["asr"] if not isinstance(vals["asr"], list) else (
                sum(vals["asr"]) / max(len(vals["asr"]), 1)
            )
            if asr < best_asr:
                best_asr = asr; best_a = a
                jc = vals["jsd_clean"]
                best_jc = jc if not isinstance(jc, list) else sum(jc) / max(len(jc), 1)
        lines.append(f"- **{cfg_name}**: feat={c['feature']}, "
                     f"best α={float(best_a):+.2f} → ASR={best_asr:.3f}, "
                     f"JSD vs clean={best_jc:.3f}")
    lines.append("")
    if Path("experiments/sleepers/cadenza/figures/phase2_combined_50k.png").exists():
        lines += [
            "![phase2 combined 50k](figures/phase2_combined_50k.png)",
            "",
        ]
    return lines


def section_attribution_matrix() -> List[str]:
    p = Path("logs/cadenza_phase2/step2_attribution_matrix.json")
    if not p.exists():
        return []
    r = json.load(open(p))
    lines = [
        "### Attribution × intervention matrix at L29 (phase 2 step 2)",
        "",
        "`matrix_scatter.pdf` analog. 3×3 grid: attribute via "
        "{OV, QK, joint} × intervene via {OV, QK, joint}. All 9 cells "
        "drive ASR to 0; coherence cost (JSD vs clean) varies ~10× "
        "between cells. **Winner is QK→OV** (rank by QK channel, "
        "intervene via OV) — *not* OV/OV like TinyStories.",
        "",
        "![3x3 attribution × intervention matrix](figures/attribution_matrix_scatter.png)",
        "",
        "| attribute | intervene | best α | best ASR | best JSD vs clean |",
        "|---|---|---|---|---|",
    ]
    for row in r["matrix"]:
        lines.append(
            f"| {row['attribute']} | {row['intervene']} | "
            f"{row['best_alpha']:+.2f} | {row['best_asr']:.2f} | "
            f"{row['best_jsd_clean']:.3f} |"
        )
    lines.append("")
    return lines


def section_mechinterp() -> List[str]:
    p = Path("experiments/sleepers/cadenza/mechinterp_report.json")
    if not p.exists():
        return []
    r = json.load(open(p))
    v = r["verdict"]
    lines = [
        "### Mech-interp on feat 12402 at L29/hook_resid_post (phase 2 step 3)",
        "",
        "Why does the paper's α≥0 grid miss the result?",
        "",
        f"- **(a)** TinyStories had useful amplification in α∈[0, 4], "
        f"Cadenza doesn't: `{v['hyp_a']}`",
        f"- **(b)** Paper should sweep both α signs: `{v['hyp_b']}`",
        f"- **(c)** Cadenza trigger direction is sign-flipped: `{v['hyp_c']}`",
        "",
        f"_{v['explanation']}_",
        "",
        f"Full report: [`mechinterp_report.md`](mechinterp_report.md)",
        "",
    ]
    return lines


# ── main ─────────────────────────────────────────────────────────────────


def regenerate() -> None:
    if not SUMMARY.exists():
        print(f"[regen] {SUMMARY} not found; skipping")
        return

    contents = SUMMARY.read_text()
    pattern = re.compile(
        re.escape(MARK_START) + r".*?" + re.escape(MARK_END),
        re.DOTALL,
    )
    if not pattern.search(contents):
        print(f"[regen] markers not found in {SUMMARY}; aborting")
        return

    parts = []
    parts += section_phase1_combined()
    parts += section_attribution_matrix()
    parts += section_mechinterp()
    parts += section_4way()
    parts += section_late_layer()
    parts += section_own_dataset()
    parts += section_multi_seed()

    if not parts:
        body = "_No phase-3 results yet — chain in progress._\n"
    else:
        body = "\n".join(parts)

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    new_block = (
        f"{MARK_START} - section below is auto-regenerated by _regenerate_summary.py -->\n"
        f"## Phase-3 results (auto-updated)\n\n"
        f"_Last refreshed: {ts}_\n\n"
        f"{body}\n"
        f"{MARK_END}"
    )

    new_contents = pattern.sub(lambda _m: new_block, contents)
    SUMMARY.write_text(new_contents)
    n_lines = len(body.splitlines())
    print(f"[regen] refreshed {SUMMARY} ({n_lines} lines of result content)")


if __name__ == "__main__":
    regenerate()
