# EM-scaling — extending the Qwen-14B QK→QK result to other model sizes

**Goal.** The paper's main-text headline (section `sec:em`,
`docs/main.tex`) shows that on Qwen2.5-14B-Instruct fine-tuned for
Emergent Misalignment (EM), an FRA **QK→QK** intervention — rescaling
top SAE features at `blocks.24.ln1.hook_normalized` — shifts the
alignment / coherence frontier substantially more than conventional
additive steering. Dmitry's framing: *"our most important result … this
has never been found before."*  This experiment generalises that result
across model sizes and benchmarks it against (i) Arditi et al.'s
SAE-DoM baseline ([LessWrong, 2025](https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG/finding-misaligned-persona-features-in-open-weight-models))
and (ii) the published EM finetune's own coherent-misalignment rate
from [Turner et al., 2025](https://arxiv.org/abs/2506.11618) (the
"Model Organisms for EM" paper, Fig 1).

## Setup

| | |
|---|---|
| Bases | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-32B-Instruct (Qwen-14B is the paper's reference cell, lives upstream in `phase1_fra_orchestrator.py`) |
| Domains | `medical`, `finance`, `sports` (the three published EM finetune flavours from `ModelOrganismsForEM/*`) |
| Recipe A — paper | **FRA QK→QK** — head-attribution sweep picks the strongest head; top-K SAE features are ranked across prompts by QK-pair attribution; intervention rescales these features at the SAE's hookpoint via encode → multiply → decode |
| Recipe B — baseline | **DoM steering** (Arditi / Turner convention) — `dom = mean(EM_act) − mean(base_act)` over a probe set; subtract `α · dom` from EM activations at the same hookpoint |
| Judge | GPT-4o paired prompts, returning alignment (0–100) and coherence (0–100). Same primitives as `phase1_judge_and_combine.py`; outputs upstream-compatible |
| Headline metric | best alignment over the α-sweep subject to coherence ≥ 70 (matches the paper convention) |
| Eval prompts | First N from `fra.em_evaluation.EM_EVAL_PROMPTS` (the 8 canonical EM-eval prompts) |
| Hardware | one NVIDIA H200 (143 GB VRAM), bf16 inference |

### Per-base configuration

| base | n_layers | default layer | hookpoint | SAE used by FRA-QK→QK |
|---|---:|---:|---|---|
| `qwen-7b`  | 28 | 15 | `hook_resid_post`     | andyrdt L15 trainer 1 (8× expansion) |
| `llama-8b` | 32 | 15 | `hook_resid_post`     | andyrdt L15 trainer 1 (8× expansion) |
| `qwen-14b` | 48 | 24 | `ln1.hook_normalized` | paper's published L24 ln1 SAE |
| `qwen-32b` | 64 | 40 | `ln1.hook_normalized` | **to be trained** (task #6) |

### Methodological caveat: SAE hookpoint mismatch

Qwen-7B and Llama-8B use `hook_resid_post` SAEs (andyrdt's are the only
public open-weight SAEs at these scales; the paper's qwen-14b SAE is at
`ln1.hook_normalized`). The QK→QK intervention at `resid_post` therefore
perturbs the residual stream that feeds **every** downstream layer,
rather than only one layer's QK/OV — a broader perturbation than the
paper's protocol. This is a real methodological caveat, not a bug, and
should be flagged when the results are interpreted: an apples-to-apples
comparison with the paper's Qwen-14B cell would require training ln1
SAEs at these scales (~3-4 hr per model on H200), which is in scope for
a follow-on if the resid_post-variant result is interesting enough.

## Pipeline

1. **`experiments/em_scaling/phase0_smoke.py`** — load each
   (base, domain) cell, attach the SAE, run a 3-prompt rollout.
2. **`experiments/em_scaling/phase1_fra_qkqk.py`** — for one
   (base, domain, eval_seed): if `default_head=0` run a quick head
   attribution sweep, rank top-K QK-pair features across the prompt set,
   sweep α ∈ {0, 0.5, 1, 1.5, 2, 3}, write
   `qualitative_FRA_<base>_<domain>_evalseed<N>.json`.
3. **`experiments/em_scaling/phase2_dom_steering.py`** — load EM,
   capture probe-set mean residual; unload; load base, capture; unload;
   `dom = mean(EM) − mean(base)`; reload EM and sweep
   α ∈ {0, 0.5, 1, 1.5, 2, 3, 4}, write
   `qualitative_DoM_<base>_<domain>_evalseed<N>.json`.
4. **`experiments/em_scaling/phase_judge.py judge ...`** — judge each
   qualitative JSON in place with GPT-4o, then
   `... combine ...` across eval seeds per cell.
5. **`reproduce/em_scaling/em_scaling_chain.sh`** — autoresearch chain
   wrapping (1)–(4) for every cell sequentially, auto-pushing partial
   progress after every cell. JSON-cache-aware: cells whose qualitative
   file already contains the full `n_prompts × (n_alphas + 1)` entries
   are skipped on re-launch, so the chain is restartable.

## Cells in scope (FRA-7 + FRA-9)

| cell | recipe coverage | status |
|---|---|---|
| `qwen-7b` × `{medical, finance, sports}` | FRA QK→QK + DoM | ⏳ in chain (seed 42) |
| `llama-8b` × `{medical, finance, sports}` | FRA QK→QK + DoM | ⏳ in chain (seed 42) |
| `qwen-14b/medical` (FRA-8) | FRA QK→QK (paper) + DoM | DoM queued via `DOM_ONLY_BASES=qwen-14b` |
| `qwen-32b/{medical, finance, sports}` | FRA QK→QK + DoM | FRA blocked on SAE training (task #6); DoM unblocks today via `DOM_ONLY_BASES=qwen-32b` |
| Cadenza Llama-3 8B sleeper (FRA-11) | DoM | driver in `experiments/sleepers/cadenza/phase2_dom.py`; pending GPU |
| TinyStories-33M sleeper (FRA-10) | DoM | driver in `experiments/sleepers/tinystories_dom.py`; ~30s end-to-end |

The first overnight chain is `BASES="qwen-7b llama-8b"` × `DOMAINS="medical
finance sports"` × `EVAL_SEEDS="42"` — six cells, ~4 hours wall-clock,
producing un-judged qualitative JSONs. Judging is deferred (no
`OPENAI_API_KEY` in this pod's `.env`); once added, run
`python -m experiments.em_scaling.phase_judge judge --qualitative
<file>` per cell, then re-run `_regenerate_summary.py` to populate the
auto-section below.

<!-- AUTO:em-scaling-start - section below is auto-regenerated by _regenerate_summary.py -->
## Results (auto-updated)

_Last refreshed: 2026-05-18 21:38 UTC_

### Phase 0 — model + SAE smoke

Per-cell loadability + a 3-prompt rollout to confirm the registry is wired correctly and the SAE attaches cleanly to the right hookpoint.

| cell | status | model_s | sae_s | fwd_ms | gen_s |
|---|---|---:|---:|---:|---:|
| `llama-8b/medical` | OK | 456.3 | 31.7 | 63.6 | 2.3 |

<!-- AUTO:em-scaling-end -->

## Reproduction

```bash
# Six-cell sweep (qwen-7b + llama-8b × medical/finance/sports, 1 eval seed):
source scripts/runpod_activate.sh   # loads .env, activates venv
bash reproduce/em_scaling/em_scaling_chain.sh

# Add qwen-14b DoM (FRA-8):
DOM_ONLY_BASES=qwen-14b bash reproduce/em_scaling/em_scaling_chain.sh

# Single cell, manual:
python -m experiments.em_scaling.phase1_fra_qkqk \
    --base qwen-7b --domain medical --eval-seed 42 \
    --output-root logs/em_scaling/phase1_fra
python -m experiments.em_scaling.phase2_dom_steering \
    --base qwen-7b --domain medical --eval-seed 42 \
    --output-root logs/em_scaling/phase2_dom

# Judge + combine (requires OPENAI_API_KEY):
python -m experiments.em_scaling.phase_judge judge \
    --qualitative logs/em_scaling/phase1_fra/qualitative_FRA_qwen-7b_medical_evalseed42.json
python -m experiments.em_scaling.phase_judge combine \
    --pattern 'logs/em_scaling/phase1_fra/gpt4o_aggregated_FRA_qwen-7b_medical_evalseed*.json' \
    --out     logs/em_scaling/phase1_fra/gpt4o_combined_FRA_qwen-7b_medical.json
python -m experiments.em_scaling._regenerate_summary
```

## Open items for Dmitry / Aniket

1. **Judge model.** Defaulting to GPT-4o for paper-comparable numbers;
   pod is missing `OPENAI_API_KEY` so judging is deferred. Once the key
   lands, the chain wrapper picks it up automatically.
2. **Qwen-32B SAE training.** ~3-4 hr on H200; adapt
   `scripts/train_topk_sae_llama.py` for Qwen-32B at L40
   `ln1.hook_normalized`, 50M Pile tokens. The Cadenza phase-3c result
   (Pile beats trigger-saturated corpora for the sleeper) suggests
   defaulting to Pile here too unless there's a specific reason to
   match the EM training corpus.
3. **Arditi comparison fidelity.** Arditi reports "fraction of coherent
   responses (coh > 50) that are misaligned (align < 30)". We report
   "alignment score @ coh ≥ 70" (paper convention). The judging primitives
   produce both; the headline metric is configurable. Decide which is
   the canonical chart for the paper revision.
4. **Three eval seeds vs one.** Current chain runs 1 eval seed for
   speed. The paper's std-across-seeds error bars want 3 seeds. Set
   `EVAL_SEEDS="42 43 44"` in the chain env to triple the run (~12 hr).
