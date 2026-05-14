#!/usr/bin/env bash
# After all 4 Phase 1 jobs (medical, random_medical, finance, sports) finish:
#   1. judge each run dir with GPT-4o
#   2. compute Δalign|coh≥70 per (em_model, condition)
#   3. plot frontier grid
#   4. push runs + plots + the published baseline to HF
#   5. exit 0 on success, 1 on failure
#
# Run from inside /workspace/fra_proj on either pod (jobs share /workspace/runs path
# but each pod owns its subset; this script aggregates whatever the local pod has).
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-/workspace/runs}"
PLOT_DIR="${PLOT_DIR:-/workspace/runs/phase1_summary}"
PUSH_HF="${PUSH_HF:-1}"

cd "$(dirname "$0")/.."
mkdir -p "$PLOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY not set. Export it then re-run."
    exit 1
fi

echo "=== 1. Judge each run dir with GPT-4o ==="
for d in "$RUNS_ROOT"/*; do
    [[ -d "$d" ]] || continue
    if compgen -G "$d/gpt4o_aggregated_*.json" >/dev/null; then
        echo "  $d already judged — skipping."
        continue
    fi
    if ! compgen -G "$d/qualitative_*.json" >/dev/null; then
        echo "  [WARN] $d has no qualitative_*.json — frontier run incomplete."
        continue
    fi
    echo "  judging $d ..."
    python3 judge_multiseed.py --results-dir "$d"
done

echo "=== 2. Aggregate + plot ==="
python3 scripts/post_phase1_analyze.py \
    --runs-root "$RUNS_ROOT" \
    --plot-out "$PLOT_DIR/frontier_grid" \
    $( [[ "$PUSH_HF" == "1" ]] && echo "--push-to-hf" )

echo "=== 3. Compare to the published's v1 baseline ==="
python3 - <<EOF
import json
from pathlib import Path

ours = json.loads(Path("$PLOT_DIR/frontier_grid.json").read_text())
orig = json.loads(Path("/workspace/fra_proj/orig_v1_baseline.json").read_text())
print()
print(f"{'em':10s} {'method':10s}  {'ours Δ':>10s}  {'orig v1 Δ':>10s}  {'gap':>8s}  {'ours peak':>10s}  {'orig peak':>10s}")
gates = []
for em in ("medical", "finance", "sports"):
    nkey = f"{em}_v1_k1"
    orig_em = orig.get(nkey, {})
    our_em  = ours.get(em, {}) or {}
    for method in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
        n = orig_em.get(method, {}) or {}
        o = our_em.get(method, {}) or {}
        nd = n.get("delta_align_coh70")
        od = o.get("delta_align_coh70")
        gap = (od - nd) if (isinstance(nd, (int, float)) and isinstance(od, (int, float))) else None
        np = n.get("peak_alignment")
        op = o.get("peak_alignment")
        ndr = "nan" if nd is None or (isinstance(nd, float) and nd != nd) else f"{nd:6.2f}"
        odr = "nan" if od is None or (isinstance(od, float) and od != od) else f"{od:6.2f}"
        gpr = "—" if gap is None else f"{gap:+6.2f}"
        npr = "—" if np is None else f"{np:6.2f}"
        opr = "—" if op is None else f"{op:6.2f}"
        print(f"{em:10s} {method:10s}  {odr:>10s}  {ndr:>10s}  {gpr:>8s}  {opr:>10s}  {npr:>10s}")
        # gate: medical qk_to_ov within 5 absolute points of the published's v1 (rough)
        if em == "medical" and method == "qk_to_ov":
            if isinstance(nd, (int, float)) and isinstance(od, (int, float)):
                ok = abs(od - nd) <= 5.0
                gates.append(("medical_qk_to_ov_within_5_of_orig_v1", ok, nd, od))
                print(f"\n  Phase 1 gate: medical QK→OV reproduction within ±5 of the published v1: {'PASS' if ok else 'FAIL'} ({od:.2f} vs {nd:.2f})")

# Save gate result for downstream
gate_out = Path("$PLOT_DIR/phase1_gate.json")
gate_out.write_text(json.dumps([
    {"name": n, "pass": ok, "orig": nd, "ours": od}
    for n, ok, nd, od in gates
], indent=2))
print(f"\nGate result → {gate_out}")
EOF

echo "=== Done. Frontier grid + summary at $PLOT_DIR ==="
