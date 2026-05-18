"""
Auto-regenerator for the EM-scaling summary.md.

Mirrors ``experiments.sleepers.cadenza._regenerate_summary``: scans every
JSON artifact produced by the EM-scaling chain (smoke aggregate, per-cell
combined judgments from phase 1 FRA QK→QK and phase 2 DoM) and rewrites
the auto-managed sections between the markers

    <!-- AUTO:em-scaling-start -->
    ...
    <!-- AUTO:em-scaling-end -->

in ``experiments/em_scaling/summary.md``.  Everything outside the marker
range is preserved verbatim — manual prose and the experimental-setup
section stay frozen.

Idempotent. Safe to call from the chain wrapper after every step.

Run::

    python -m experiments.em_scaling._regenerate_summary
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SUMMARY = Path("experiments/em_scaling/summary.md")
MARK_START = "<!-- AUTO:em-scaling-start"
MARK_END = "<!-- AUTO:em-scaling-end -->"

BASES = ("qwen-7b", "llama-8b", "qwen-14b", "qwen-32b")
DOMAINS = ("medical", "finance", "sports")


def _load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _combined_for(recipe: str, base: str, domain: str) -> dict | None:
    root = Path(f"logs/em_scaling/phase{1 if recipe == 'FRA' else 2}_"
                f"{'fra' if recipe == 'FRA' else 'dom'}")
    candidates = sorted(root.glob(
        f"gpt4o_combined_{recipe}_{base}_{domain}.json"))
    if not candidates:
        return None
    return _load_json(candidates[0])


def section_smoke() -> List[str]:
    p = Path("logs/em_scaling/smoke/smoke_aggregate.json")
    if not p.exists():
        return []
    rows = _load_json(p) or []
    if isinstance(rows, dict):
        rows = rows.get("cells", []) or list(rows.values())
    if not rows:
        return []
    lines = [
        "### Phase 0 — model + SAE smoke",
        "",
        "Per-cell loadability + a 3-prompt rollout to confirm the registry "
        "is wired correctly and the SAE attaches cleanly to the right hookpoint.",
        "",
        "| cell | status | model_s | sae_s | fwd_ms | gen_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        cell = f"{r.get('base','?')}/{r.get('domain','?')}"
        st = r.get("status", "?")
        ms = r.get("t_model_load_s", r.get("model_s", "-"))
        ss = r.get("t_sae_load_s", r.get("sae_s", "-"))
        fw = r.get("single_fwd_ms", r.get("fwd_ms", "-"))
        gs = r.get("batched_gen_s", r.get("gen_s", "-"))
        lines.append(f"| `{cell}` | {st} | {ms} | {ss} | {fw} | {gs} |")
    lines.append("")
    return lines


def _winner(combined: dict, coh_floor: float = 70.0) -> dict | None:
    """Pick the method × α that maximises alignment subject to coh ≥ floor."""
    best = None
    for method, body in combined.get("by_method", {}).items():
        for e in body["by_alpha"]:
            if e.get("mean_coherence_across_seeds", e.get("mean_coherence", 0)) >= coh_floor:
                al = e.get("mean_alignment_across_seeds", e.get("mean_alignment"))
                if best is None or al > best["alignment"]:
                    best = {
                        "method": method, "scale": e["scale"],
                        "alignment": al,
                        "coherence": e.get("mean_coherence_across_seeds",
                                           e.get("mean_coherence")),
                        "n_seeds": e.get("n_seeds", e.get("n", 1)),
                    }
    return best


def _baseline(combined: dict) -> dict | None:
    body = combined.get("by_method", {}).get("baseline")
    if not body:
        return None
    e = body["by_alpha"][0] if body["by_alpha"] else None
    if e is None:
        return None
    return {
        "alignment": e.get("mean_alignment_across_seeds", e.get("mean_alignment")),
        "coherence": e.get("mean_coherence_across_seeds", e.get("mean_coherence")),
        "n_seeds":   e.get("n_seeds", e.get("n", 1)),
    }


def section_headline_table() -> List[str]:
    """For every (base, domain) we have a combined JSON, list FRA / DoM /
    baseline best-alignment @ coh≥70."""
    rows = []
    for base in BASES:
        for domain in DOMAINS:
            fra = _combined_for("FRA", base, domain)
            dom = _combined_for("DoM", base, domain)
            if fra is None and dom is None:
                continue
            fra_w = _winner(fra) if fra else None
            dom_w = _winner(dom) if dom else None
            base_a = (_baseline(fra) if fra else None) or (
                _baseline(dom) if dom else None)
            rows.append((base, domain, fra_w, dom_w, base_a))

    if not rows:
        return []

    lines = [
        "### Headline: alignment @ coh ≥ 70 per cell",
        "",
        "For each (base, domain) cell we report the best alignment "
        "achievable at coherence ≥ 70 under each recipe, alongside the "
        "no-steer baseline. Higher = more aligned; the gain over baseline "
        "is the steering's contribution to the EM frontier.",
        "",
        "| base | domain | baseline | FRA QK→QK (α*) | DoM (α*) | Δ FRA−DoM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for base, domain, fra_w, dom_w, b in rows:
        base_s = f"{b['alignment']:.1f}" if b else "—"
        fra_s = (f"{fra_w['alignment']:.1f} (α={fra_w['scale']:+.2f}, "
                 f"{fra_w['method']})") if fra_w else "—"
        dom_s = (f"{dom_w['alignment']:.1f} (α={dom_w['scale']:+.2f})"
                 ) if dom_w else "—"
        if fra_w and dom_w:
            delta = fra_w["alignment"] - dom_w["alignment"]
            delta_s = f"{delta:+.1f}"
        else:
            delta_s = "—"
        lines.append(f"| {base} | {domain} | {base_s} | {fra_s} | {dom_s} | {delta_s} |")
    lines.append("")
    return lines


def section_per_cell_sweeps() -> List[str]:
    """One small block per cell showing the FRA QK→QK and DoM α-sweeps."""
    lines: List[str] = []
    for base in BASES:
        for domain in DOMAINS:
            fra = _combined_for("FRA", base, domain)
            dom = _combined_for("DoM", base, domain)
            if fra is None and dom is None:
                continue
            lines += [
                f"#### `{base}` / `{domain}`",
                "",
            ]
            for label, combined in (("FRA QK→QK", fra), ("DoM", dom)):
                if combined is None:
                    continue
                lines.append(
                    f"**{label}** "
                    f"(L{combined.get('layer','?')} "
                    f"H{combined.get('head','?')})"
                )
                lines.append("")
                lines.append("| method | α | alignment | coherence | n_seeds |")
                lines.append("|---|---:|---:|---:|---:|")
                for method, body in combined["by_method"].items():
                    for e in body["by_alpha"]:
                        a = e.get("mean_alignment_across_seeds",
                                  e.get("mean_alignment"))
                        c = e.get("mean_coherence_across_seeds",
                                  e.get("mean_coherence"))
                        n = e.get("n_seeds", e.get("n", 1))
                        lines.append(f"| `{method}` | {e['scale']:+.2f} | "
                                     f"{a:.1f} | {c:.1f} | {n} |")
                lines.append("")
    return lines


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

    parts: List[str] = []
    parts += section_smoke()
    parts += section_headline_table()
    parts += section_per_cell_sweeps()

    if not parts:
        body = "_No EM-scaling results yet — chain in progress._\n"
    else:
        body = "\n".join(parts)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new_block = (
        f"{MARK_START} - section below is auto-regenerated by "
        f"_regenerate_summary.py -->\n"
        f"## Results (auto-updated)\n\n"
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
