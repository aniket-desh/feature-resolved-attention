#!/bin/bash
# Autonomous v1 → v2 handoff: wait for the v1 sweep PID to exit, dump a
# summary, then sequentially launch the v2 sweep over all 9 cells using
# the SAEs trained during v1. No user input required.
#
# Usage (called from inside this repo):
#   bash reproduce/sleepers/cadenza_autonomous_handoff.sh <v1_pid>
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

V1_PID=${1:?must pass v1 sweep pid as $1}
PY=${PY:-/usr/bin/python3}

echo "[handoff] waiting for v1 sweep pid=$V1_PID to exit..."
while kill -0 "$V1_PID" 2>/dev/null; do
  sleep 60
done
echo "[handoff] v1 done at $(date)"

echo
echo "================================================================"
echo "  v1 SUMMARY"
echo "================================================================"
"$PY" -c "
import json, glob, pathlib
rows = []
for p in sorted(glob.glob('logs/cadenza_localisation/cadenza_L*.json')):
    r = json.load(open(p))
    rows.append({
        'cell': pathlib.Path(p).stem,
        'feat': r['selection']['feature'],
        'a': r['selection']['alpha'],
        'val_asr': r['selection']['val_asr'],
        'val_dce': r['selection']['val_delta_ce'],
        'test_asr': r['test']['asr'],
        'test_dce': r['test']['delta_ce'],
    })
print(f'{\"cell\":<46} {\"feat\":>7} {\"α*\":>5} {\"vASR\":>5} {\"vΔCE\":>8} {\"tASR\":>5} {\"tΔCE\":>8}')
for r in rows:
    print(f'{r[\"cell\"]:<46} {r[\"feat\"]:>7} {r[\"a\"]:+.2f} {r[\"val_asr\"]:>5.2f} {r[\"val_dce\"]:+8.4f} {r[\"test_asr\"]:>5.2f} {r[\"test_dce\"]:+8.4f}')
print()
zeros = [r for r in rows if r['test_asr'] == 0.0]
print(f'cells with test ASR=0 (v1): {len(zeros)}')
" | tee logs/cadenza_localisation/v1_summary.txt

echo
echo "================================================================"
echo "  LAUNCHING v2 SWEEP"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
v2_log="logs/cadenza_localisation_v2/sweep_${ts}.log"
mkdir -p "$(dirname "$v2_log")"

bash reproduce/sleepers/cadenza_v2_sweep.sh > "$v2_log" 2>&1
echo "[handoff] v2 sweep finished at $(date); log: $v2_log"

echo
echo "================================================================"
echo "  v2 SUMMARY"
echo "================================================================"
"$PY" -c "
import json, glob, pathlib
rows = []
for p in sorted(glob.glob('logs/cadenza_localisation_v2/cadenza_L*.json')):
    r = json.load(open(p))
    rows.append({
        'cell': pathlib.Path(p).stem,
        'feat': r['selection']['feature'],
        'a': r['selection']['alpha'],
        'val_asr': r['selection']['val_greedy_asr'],
        'val_dce': r['selection']['val_delta_ce'],
        'test_asr': r['test']['asr_mean'],
        'test_dce': r['test']['delta_ce'],
        'per_seed': r['test']['per_seed_asr'],
    })
print(f'{\"cell\":<46} {\"feat\":>7} {\"α*\":>5} {\"vASR\":>5} {\"vΔCE\":>8} {\"tASR\":>5} {\"tΔCE\":>8}')
for r in rows:
    print(f'{r[\"cell\"]:<46} {r[\"feat\"]:>7} {r[\"a\"]:+.2f} {r[\"val_asr\"]:>5.2f} {r[\"val_dce\"]:+8.4f} {r[\"test_asr\"]:>5.2f} {r[\"test_dce\"]:+8.4f}')
print()
zeros = [r for r in rows if r['test_asr'] == 0.0]
print(f'cells with mean test ASR=0 (v2): {len(zeros)}')
low = [r for r in rows if r['test_asr'] <= 0.2]
print(f'cells with mean test ASR ≤ 0.2 (v2): {len(low)}')
for r in low:
    print(f'  → {r[\"cell\"]}  feat={r[\"feat\"]}  α={r[\"a\"]:+.2f}  '
          f'tASR={r[\"test_asr\"]:.2f}  per_seed={r[\"per_seed\"]}')
" | tee logs/cadenza_localisation_v2/v2_summary.txt

echo
echo "[handoff] DONE  $(date)"
