# EM-scaling handoff doc

Written for the next Claude Code instance picking up this work. Read end-to-end
before doing anything — there's specific Dmitry context, codebase patterns,
and a GPU-stack blocker that affect the right next steps.

---

## What this experiment is

**Goal:** Extend the paper's main-text QK→QK steering result on Qwen2.5-14B
to other open-weight model sizes, then benchmark against three baselines.

The original paper (`docs/main.tex`, section `sec:em`) reports that on
Qwen2.5-14B-Instruct fine-tuned for emergent misalignment (EM), a FRA QK→QK
intervention (rescaling top SAE features at `blocks.24.ln1.hook_normalized`)
shifts the alignment–coherence frontier substantially more than conventional
additive steering. Dmitry's message naming this as "our most important result"
and the next step:

> Our most important result is the superior EM steering via the QK/QK. This is
> very surprising and, AFAIK, has never been found before. I'm scaling this to
> Qwen-32B in the neel nanda paper. Would be good to do the others as well
> (Gemma, 7B). We should compare to the arditi results in
> [Finding Misaligned Persona Features in Open-Weight Models]
> (https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG/...).
> In particular I want to compare to the arditi result in 7B.
> This may already be true, but one thing that would be big if true is if we can
> get superior EM rate @ coh 70 to the finetunes here.
> Would also be good to benchmark against DoM steering.

Concrete deliverables:

1. FRA QK→QK on **Qwen2.5-7B** × {medical, finance, sports} — to compare
   directly against Arditi.
2. FRA QK→QK on **Llama-3.1-8B-Instruct** × {medical, finance, sports} —
   Arditi also tested it.
3. FRA QK→QK on **Qwen2.5-32B-Instruct** × {medical, finance, sports} —
   the Neel Nanda "Model Organisms for Emergent Misalignment" target.
4. **DoM steering** as a baseline on the same cells (compute mean
   activation diff between EM model and base, subtract scaled multiple).
5. Compare against the **published finetune's** baseline EM rate @ coh ≥ 70.
6. Compare against **Arditi's reported numbers** at matched cells
   (qualitative scatter overlay — they report on Qwen-7B + Llama-8B medical).

Dmitry's framing: "Gemma" is mentioned but Arditi doesn't test Gemma, and
there's no `ModelOrganismsForEM/Gemma-*` finetune. **Skip Gemma for now**
unless a Gemma EM model surfaces.

---

## External references the next instance should keep in mind

| Reference | What's relevant |
|---|---|
| `docs/main.tex` (this repo) | The original FRA paper. Section `sec:em` and appendix `app:matrix` are the QK→QK protocol you're scaling. |
| [Arditi et al. — "Finding Misaligned Persona Features"](https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG/finding-misaligned-persona-features-in-open-weight-models) | Tests Llama-3.1-8B + Qwen2.5-7B with **SAE-DoM** steering on EM finetunes. Misalignment rate = fraction of coherent (coh > 50) responses that are misaligned (align < 30). Code: https://github.com/safety-research/open-source-em-features. |
| [Turner et al. — Model Organisms for EM](https://arxiv.org/abs/2506.11613) | Source of `ModelOrganismsForEM/*` HF org. ~40% misalignment on Qwen-32B at >99% coherence. |
| `experiments/sleepers/cadenza/summary.md` (this repo) | The other autoresearch we ran end-to-end. Read it to absorb the **autoresearch chain** pattern, the regen-summary-from-JSON pattern, and the auto-push wiring. |

---

## State of the world right now (2026-05-18 ~03:30 UTC)

**Built and committed (push `a5c490d`):**
- `fra/em_models.py` — registry + loaders for qwen-7b, qwen-14b, qwen-32b,
  llama-8b. SAE picks per base are concrete.
- `experiments/em_scaling/phase0_smoke.py` — per-cell load + 3-prompt rollout.
- `experiments/em_scaling/phase1_fra_qkqk.py` — generalised FRA QK→QK driver
  (one (base, domain) cell at a time).
- `experiments/em_scaling/phase2_dom_steering.py` — DoM steering baseline
  (load EM, capture probe activations, unload; load base, capture, unload;
  compute DoM; reload EM and α-sweep).

**Not yet built (do these next, in this order):**
1. `experiments/em_scaling/_regenerate_summary.py` — mirrors the Cadenza version
   in `experiments/sleepers/cadenza/_regenerate_summary.py`. Scans all JSON
   artifacts and refreshes a "Phase results" section in
   `experiments/em_scaling/summary.md` between AUTO markers.
2. `experiments/em_scaling/summary.md` — skeleton with AUTO marker block.
   Have it explain the experimental setup (mostly cribbed from this handoff).
3. `reproduce/em_scaling/em_scaling_chain.sh` — autoresearch chain wrapper.
   For each (base, domain) in the 9-cell grid: run phase 1, judge it, run
   phase 2, judge it, regen summary, auto-push.
4. `scripts/auto_push_em_scaling.sh` — same pattern as
   `scripts/auto_push_results.sh` (the Cadenza version), staging only the
   em_scaling artifacts.
5. **Qwen-32B SAE training** — no clean public SAE for Qwen2.5-32B-Instruct.
   Adapt `scripts/train_topk_sae_llama.py` to handle Qwen-32B (different
   d_model=5120 etc.). Train at L40 ln1.hook_normalized (mid-network of 64),
   50M tokens. ~3-4 hr on H200. Required before phase-3-style runs on 32B.
6. `experiments/em_scaling/_make_summary_plots.py` — plots come *after*
   you have results. Probably 4-5 useful plots (see "Useful plots" below).

**Blocked on (read this carefully):**

When I tried to launch `phase0_smoke.py` it failed with:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

I confirmed via clean Python processes that **even `torch.zeros(1, device='cuda')`
fails**. `nvidia-smi` reports the H200 as healthy (0 MiB used, no processes,
ECC 0/0, normal clocks). `nvidia-smi --gpu-reset` returns "Insufficient
Permissions". This is a system-level GPU stack failure I can't fix from
inside the container.

**Aniket said he would restart the pod.** When you read this, the pod has
either been restarted (CUDA works) or it hasn't (re-test with a basic
`torch.zeros(1, device='cuda')` first).

If CUDA still doesn't work: ping Aniket and stop here.

If CUDA works: run phase-0 smoke as described in "How to resume" below.

---

## Architectural choices and gotchas

### Model registry (`fra/em_models.py`)

The registry's `default_layer` and SAE choice for each base:

| base | n_layers | default layer | hookpoint | SAE | notes |
|---|---|---|---|---|---|
| qwen-7b | 28 | 15 | hook_resid_post | andyrdt L15 trainer 1 | midpoint-aligned |
| qwen-14b | 48 | 24 | ln1.hook_normalized | paper's published SAE | reference cell |
| qwen-32b | 64 | 32 | ln1.hook_normalized | **needs training** | mid-network |
| llama-8b | 32 | 15 | hook_resid_post | andyrdt L15 trainer 1 | midpoint |

**Gotcha 1: hookpoint mismatch with paper.** The paper does QK→QK at
`ln1.hook_normalized`. The andyrdt SAEs are at `hook_resid_post`. The
generalised driver picks the hookpoint per `sae_kind`:
- `qwen_ln1` (Qwen-14B paper SAE) → ln1.hook_normalized
- `qwen_resid` (andyrdt) → hook_resid_post
- `local` (Qwen-32B you'll train) → ln1.hook_normalized

This means the **Qwen-7B and Llama-8B** runs are a slight variant of the
paper protocol — the intervention modifies the residual stream that feeds
*every downstream layer* rather than just one layer's QK/OV. Mention this
explicitly in the summary.md when results land — it's a real
methodological caveat, not a bug.

If/when someone trains ln1-hookpoint SAEs for these models, switch over
for an apples-to-apples paper comparison.

**Gotcha 2: head selection.** The paper uses head 38 on Qwen-14B based on
`head_attribution_sweep`. For other bases the registry has
`default_head=0`, and `phase1_fra_qkqk.py` runs the ablation online when
the head is 0. The ablation only uses 3 prompts by default — should be
enough, but the picked head will vary per (base, domain). Record it in the
JSON output and verify it's not pathological (head 0, which is sometimes a
"attend-everywhere" head, would be a red flag).

**Gotcha 3: low_cpu_mem_usage in `from_pretrained` was the bug source.**
I initially had `low_cpu_mem_usage=True` and it caused the
HookedTransformer wrap step to fail (you saw "CUDA busy" errors that
weren't actually CUDA's fault — they were artifacts of the meta-tensor
load not materialising properly). The committed version removes it. Don't
add it back.

### Auto-push pattern (already wired for Cadenza)

There's a memory entry in
`~/.claude/projects/-workspace-aniket-feature-resolved-attention/memory/feedback_auto_push_summary.md`
saying "auto-commit and push summary.md updates without asking during
autoresearch runs". Honour that for em_scaling too. Use the same
token-auth pattern:

```bash
git push "https://aniket-desh:${GH_TOKEN}@github.com/aniket-desh/feature-resolved-attention.git" main \
    2>&1 | sed "s|${GH_TOKEN}|<token>|g"
```

The Cadenza chain has a working helper at `scripts/auto_push_results.sh`.
**Make an em_scaling-specific copy** (`scripts/auto_push_em_scaling.sh`)
that stages only `experiments/em_scaling/`, `logs/em_scaling/`,
`fra/em_models.py`. Don't `git add` indiscriminately — `uv.lock` is out
of sync with pyproject.toml; `logs/sae_training/` has multi-megabyte
training logs.

### Single-GPU vectorisation strategy

You have **one H200, 143 GB VRAM**. The constraint is sequential per model.
Within a model, vectorise via:

- `fra.em_evaluation.generate_with_hooks_batch` — batches multiple prompts
  × seeds × α into one forward pass. Already used in `phase1_fra_qkqk.py`.
- For DoM: load EM, capture all probe activations, then load base, capture
  same, then compute DoM offline, then reload EM. (Already in
  `phase2_dom_steering.py`.)
- For Qwen-32B (64 GB bf16): a single instance + SAE + tensors ≈ 95 GB.
  Use small batch sizes (1-4) and short context. **Do not load the base
  model + EM model simultaneously for the DoM phase** — sequential is fine.

---

## How to resume (when CUDA is back)

```bash
# 0. Confirm CUDA works (must succeed before anything else)
python -c "import torch; x=torch.zeros(1, device='cuda'); print(x)"

# 1. Smoke: load Qwen-7B + Llama-8B + Qwen-32B, one cell each (~30 min total)
python -m experiments.em_scaling.phase0_smoke \
    --bases qwen-7b llama-8b qwen-32b \
    --domains medical --n-prompts 3 --max-new-tokens 64

# Expect: qwen-7b / llama-8b succeed; qwen-32b skipped because its SAE
# isn't trained yet (registry's sae_kind="local", sae_release="").

# 2. Once smoke passes for at least qwen-7b, run phase 1 for one cell:
python -m experiments.em_scaling.phase1_fra_qkqk \
    --base qwen-7b --domain medical --eval-seed 42 \
    --output-root logs/em_scaling/phase1_fra

# Output: logs/em_scaling/phase1_fra/qualitative_FRA_qwen-7b_medical_evalseed42.json
# Schema is upstream-compatible with `phase1_judge_and_combine.py`.

# 3. Judge that JSON (GPT-4o; needs ANTHROPIC_API_KEY / OPENAI_API_KEY in .env)
python phase1_judge_and_combine.py \
    --qualitative logs/em_scaling/phase1_fra/qualitative_FRA_qwen-7b_medical_evalseed42.json \
    --out logs/em_scaling/phase1_fra/judged_FRA_qwen-7b_medical_evalseed42.json
# (Inspect phase1_judge_and_combine.py for its exact CLI — Aniket's
#  current Cadenza run didn't use it; the existing 14B EM pipeline does.)

# 4. Phase 2 DoM on same cell:
python -m experiments.em_scaling.phase2_dom_steering \
    --base qwen-7b --domain medical --eval-seed 42 \
    --output-root logs/em_scaling/phase2_dom

# 5. Once one (base, domain) cell works end-to-end with both recipes,
#    you'll know your time budget and can build the wrapper chain and
#    queue all 9 cells autonomously overnight.
```

---

## Useful plots (build after results land)

Mirror the Cadenza pattern of `_make_summary_plots.py` + auto-included
images in summary.md. Suggested:

1. **Alignment–coherence frontier per (base, domain)** — one panel per
   cell, FRA QK→QK vs DoM vs no-steer baseline. Y = alignment, X =
   coherence, color = recipe, point per α.
2. **Δ alignment @ coh ≥ 70 bars** — one bar per (base, domain, recipe).
   Direct comparison: FRA QK→QK should clear DoM if Dmitry's hypothesis
   is right. Compare against published Arditi numbers as a horizontal
   line where the cells match.
3. **Scaling plot** — Δ alignment @ coh ≥ 70 against model parameter
   count (4B → 7B → 8B → 14B → 32B). Shows whether the QK→QK advantage
   over DoM grows / shrinks / saturates with scale.
4. **Per-α heatmap** of alignment for one representative cell — visualises
   the α sweep finely.
5. **Comparison vs published finetune baselines** — small table or bars
   showing "what the model was before steering" vs "what we got via QK→QK".

---

## Open questions you might need to escalate to Aniket / Dmitry

1. **Hookpoint canonicalisation.** Should we train ln1 SAEs on Qwen-7B
   and Llama-8B to match the paper protocol exactly, or accept the
   resid_post variant since it's the only thing andyrdt published? My
   guess: Aniket would prefer the resid_post variant first (faster), then
   train ln1 SAEs if the resid_post result is interesting enough to
   warrant a paper-clean follow-up.
2. **Qwen-32B SAE training.** ~3-4 hr on H200. Acceptable. But before
   training, confirm the dataset choice with Aniket — Pile vs synthetic
   trigger mix vs Cadenza-style "model's own corpus". Cadenza established
   that **Pile beats trigger-saturated corpora** (phase 3c) on the
   sleeper task. Default to Pile.
3. **Arditi comparison fidelity.** Their measurement is "fraction of
   coherent responses (coh > 50) that are misaligned (align < 30)".
   The paper uses "alignment score @ coh ≥ 70". Different metric, same
   judging axis. To match Arditi exactly, compute both. Decide with
   Aniket which is the "headline" number.

---

## Codebase pointers for orienting yourself

The repo has both the **paper-bundle code** (`fra/core/`, `fra/em_evaluation.py`,
phase0_*.py / phase1_*.py at repo root) and the **two autoresearch extensions
we built**:

| Path | Contents |
|---|---|
| `fra/core/{fra,ov,helpers}.py` | Model-agnostic FRA tensor code. Reused everywhere. |
| `fra/em_evaluation.py` | 2036 LOC — EM eval prompts, GPT-4o judge, batched generation. **The `generate_with_hooks_batch` and `rank_features_multi_prompt` functions are the canonical primitives.** |
| `phase1_fra_orchestrator.py` (repo root) | The paper's Qwen-14B FRA driver. **Read this first** when generalising — every pattern in `experiments/em_scaling/phase1_fra_qkqk.py` came from here. |
| `phase1_additive_orchestrator.py` | Same shape, but additive (paper's conventional baseline). |
| `phase1_judge_and_combine.py` | Judging stage. Consumes qualitative JSON, outputs judged JSON. **Don't reinvent — use it.** |
| `fra/em_models.py` | NEW. The registry. |
| `fra/sae_lens_wrapper.py` | `QwenSAE`, `QwenLn1SAE`, `LocalLn1SAE`, `GemmaScopeSAE`. The wrappers expose a uniform interface (`encode`, `decode`, `W_dec`, `W_enc`, `b_dec`, `b_enc`, `layer`, `d_in`, `d_sae`). |
| `fra/head_ablation.py` | `head_attribution_sweep` — used by phase 1 when default_head==0. |
| `experiments/sleepers/cadenza/` | Reference autoresearch. **Read `summary.md` + `_regenerate_summary.py` + `_make_summary_plots.py` + the auto-push chain in `reproduce/sleepers/cadenza_phase2_chain.sh`.** Every pattern repeats here. |
| `reproduce/sleepers/cadenza_phase2_chain.sh` | Reference chain wrapper. The em_scaling chain should look identical in shape. |
| `scripts/auto_push_results.sh` | Reference auto-pusher. Make a copy targeting em_scaling artifacts. |

---

## TaskList state when I left

```
#20 [in_progress] EM-scaling phase 0: model + SAE smoke for Qwen-7B/Llama-8B/Qwen-32B
                  Failed twice with CUDA-busy. Awaiting pod restart.
#21 [completed]   Build fra/em_models.py model registry + generalized loader
#22 [completed]   Build phase1_fra_qkqk.py (generalised across model size)
#23 [completed]   Build phase2_dom_steering.py (DoM steering baseline)
#24 [pending]     Train Qwen2.5-32B-Instruct SAE
#25 [pending]     EM-scaling autoresearch chain + auto-summary
```

When you pick up, mark #20 in_progress, re-run the smoke, and if it
passes, mark it complete and start #25 (the chain). #24 (Qwen-32B SAE
training) can be parallel-launched once the chain is running for 7B/8B.

---

## One final note

Aniket's not the primary author on this paper — he's a secondary
contributor doing follow-on work. Dmitry is the main contributor. So
when results land:

1. Push to the repo as you go (memory entry permits this).
2. After a coherent batch of results lands (e.g. one model fully done
   across 3 domains, or DoM-vs-FRA scatter for one model), draft Aniket
   a Slack message he can paste to Dmitry summarising what's new.
3. The summary should *match the paper's voice* — intuitive, not
   number-heavy in chat; deep numbers live in summary.md.

Good luck.
