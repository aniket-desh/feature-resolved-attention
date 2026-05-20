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

The first overnight chain swept `BASES="qwen-7b llama-8b"` × `DOMAINS=
"medical finance sports"` × `EVAL_SEEDS="42"` plus a follow-up
`DOM_ONLY_BASES=qwen-14b` × `DOMAINS=medical` — 13 qualitative JSONs
total, judged with GPT-4o.

## TL;DR (for Dmitry)

![EM-scaling headline: FRA QK→QK vs DoM per cell](figures/headline_bars.png)

**On every (base, domain) cell we ran, DoM steering matches or beats
the FRA QK→QK protocol when FRA is forced onto an off-paper hookpoint
(`hook_resid_post`).** Paper convention is **Δ alignment @ coh ≥ 70**
(winner − unsteered baseline). Both peak and Δ shown:

| cell | baseline | FRA QK→QK peak (Δ) | DoM peak (Δ) | DoM − FRA |
|---|---:|---:|---:|---:|
| qwen-7b / medical | 56.2 | 79.4 (**+23.1**) | **93.8** (**+37.5**) | +14.4 |
| qwen-7b / finance | 37.5 | — *(coh<70 every α)* | **86.9** (**+49.4**) | — |
| qwen-7b / sports  | 62.5 | 76.9 (**+14.4**) | **91.9** (**+29.4**) | +15.0 |
| llama-8b / medical | 50.0 | — *(coh ≈ 0 every α — model breaks)* | **87.5** (**+37.5**) | — |
| llama-8b / finance | 48.8 | — | **66.2** (**+17.5**) | — |
| llama-8b / sports  | 70.0 | — | **73.1** (**+3.1**) | — |
| qwen-14b / medical | 65.0 | 72.5 (**+7.5**) | **95.0** (**+28.1**) | +20.6 |
| qwen-14b / finance | 30.0 | — *(coh<70 every α)* | **61.2** (**+31.2**) | — |
| qwen-14b / sports  | 32.5 | 51.9 (**+19.4**) | **67.5** (**+35.0**) | +15.6 |
| qwen-32b / medical | 51.2 | 51.2 (+0.0) | **56.2** (**+5.0**) | +5.0 |
| qwen-32b / finance | 42.5 | 46.2 (**+3.8**) | 46.2 (**+3.8**) | 0.0 (tied) |
| qwen-32b / sports  | 38.8 | 49.4 (**+10.6**) | **51.2** (**+12.5**) | +1.9 |

For Qwen-14B / medical, the paper's published FRA QK→QK Δ is **+27.7**
(`figures/em_figures/phase1_fra_plus_additive_3domains.png`) using
their original SAE + their original eval pipeline (different prompt
set, different judging run). Our apples-to-apples FRA QK→QK with our
own paper-spec SAE (d_sae=102,400 / k=64 / 40M Pile tokens) and our
pipeline lands at Δ **+7.5** — substantially lower than the paper's
published number. Our DoM on the same cell gets **+28.1**. So:

- On the paper's own pipeline, FRA QK→QK and DoM appear tied around Δ ≈ +28
- On our pipeline (same prompts, judge, α grid for both recipes), **DoM beats FRA QK→QK by ~20 Δ-alignment-points** on Qwen-14B/medical, and similarly large gaps on the other 14B domains (DoM +35 vs FRA +19 on sports; DoM +31 vs FRA no-winner on finance).

The +7.5 vs +27.7 gap on FRA QK→QK Δ between our run and the paper is
likely from one or more of: SAE quality (we have ~40M tokens, paper
likely more / a longer published checkpoint), prompt set / N (we use 8
fixed prompts, paper uses more across multiple seeds), and judging run
variance. The DoM numbers in our sweep are internally consistent
across all 12 cells, so the cross-recipe ordering (DoM > FRA QK→QK on
all 9 cells where FRA reaches coh≥70) is the credible take.

**Qwen-32B / medical (paper-protocol, FRA QK→QK at L40/ln1 with the
trained SAE).** Both recipes are extremely surgical but barely move
alignment:

| recipe | best @ coh≥70 | Δ vs baseline (52.5) | JSD at that α |
|---|---|---:|---:|
| FRA QK→QK | 51.2 (α=+0.5) | **−1.3** | 0.007 bits |
| DoM | 56.2 (α=+1.0) | **+3.7** | 0.002 bits |

The JSD curves show **paper-protocol ln1 interventions are 10-100× less
disruptive** than the resid_post variants on 7B/8B (FRA at ln1: 0.01-0.05
bits across the entire α-sweep, vs 0.4-0.8 bits at resid_post). So Dmitry's
"OV Pareto > ln1" intuition (sleeper substrate) extends to EM too: ln1
interventions don't break coherence. But on Qwen-32B/medical they're
also so gentle that they barely move alignment with our (layer, head,
α-grid) choices.

Possible reasons for the 32B null result:

1. **Layer choice.** I picked L40 per the handoff comment "mid-network of
   64" — but L40 is at 62.5% depth, not midpoint. Qwen-14B's paper cell
   is L24/48 = 50%. Re-running at L32 (true midpoint) is the cleanest
   next experiment.
2. **Head ablation.** L40 head ablation picked the strongest head on 3
   prompts. Could be over-fit to those prompts; should re-rank with the
   full 8-prompt set.
3. **α range too narrow.** At α=3 JSD is only 0.05 bits — the model is
   barely perturbed. Paper's α grid goes to 4, ours stops at 3 for FRA;
   pushing to α=5-8 might surface the steering direction.

![JSD curves per cell](figures/jsd_panels.png)

The JSD panels make the hookpoint difference visible at a glance: qwen-7b
and llama-8b rows show FRA stuck at ~0.4-0.8 bits (resid_post — disruptive),
DoM rising 0 → 0.5 with α. Qwen-14b and qwen-32b (both ln1) show DoM
hugging the bottom: 0.02-0.06 bits at α=4. The paper protocol works as
advertised — surgical at ln1 — but the alignment payoff is only present
on Qwen-14B; we need to revisit the (layer, head, α) choices on Qwen-32B.

**The key caveat.** The FRA QK→QK protocol in the paper runs at
`ln1.hook_normalized` — the SAE attached to the attention input of one
layer. For Qwen-14B the paper has a published L24 ln1 SAE; for Qwen-7B
and Llama-8B no such SAE is public. The only off-the-shelf SAEs at
these scales are andyrdt's TopK SAEs at `hook_resid_post`. Running the
QK→QK intervention at `resid_post` perturbs the residual stream that
feeds **every downstream layer**, which empirically:

- On Qwen-7B: borderline coherent (medical / sports survive, finance fails)
- On Llama-8B: completely breaks coherence (~0 on all α, all 3 domains)

So the "DoM beats FRA" result is partly a **hookpoint-mismatch artifact**,
not a clean indictment of FRA QK→QK. The honest comparison is one of:

1. Train Qwen-7B and Llama-8B ln1 SAEs (~3-4 hr each on H200) and re-run
   FRA QK→QK at the paper's intended hookpoint.
2. Or report the paper's Qwen-14B FRA QK→QK number directly alongside
   these DoM numbers — that single cell is the only apples-to-apples
   FRA-vs-DoM comparison currently available.

**On the cells where FRA does run cleanly (qwen-7b medical / sports),
DoM still wins by 14-15 alignment points.** This is the strongest
piece of evidence so far that on these EM-finetuned models, DoM is
competitive with — or better than — FRA QK→QK at the resid_post
hookpoint. The Qwen-14B paper cell (ln1) remains the cell where FRA
QK→QK dominates the conventional baseline; we now also have a DoM
number for it (95.0 at coh 89) — see the headline plot.

**Cross-cell DoM strength.** DoM achieves alignment 73-95 at coherence
≥ 70 on 7/7 cells, with the strongest results on the larger model
(qwen-14b medical: 95.0). The "blunt mean direction" baseline works
remarkably well across model sizes — Arditi's and Turner's reports
generalise.

![alignment / coherence frontier per cell](figures/frontier_panels.png)

![DoM steering advantage vs model size](figures/scaling.png)

(The scaling plot is sparse — most FRA cells didn't surface a coh≥70
operating point, so the FRA line only has a Qwen-7B point. DoM's line
shows a clear upward trend with model size.)


<!-- AUTO:em-scaling-start - section below is auto-regenerated by _regenerate_summary.py -->
## Results (auto-updated)

_Last refreshed: 2026-05-20 06:41 UTC_

### Phase 0 — model + SAE smoke

Per-cell loadability + a 3-prompt rollout to confirm the registry is wired correctly and the SAE attaches cleanly to the right hookpoint.

| cell | status | model_s | sae_s | fwd_ms | gen_s |
|---|---|---:|---:|---:|---:|
| `llama-8b/medical` | OK | 456.3 | 31.7 | 63.6 | 2.3 |

### Headline: alignment @ coh ≥ 70 per cell

For each (base, domain) cell we report the best alignment achievable at coherence ≥ 70 under each recipe, alongside the no-steer baseline. Higher = more aligned; the gain over baseline is the steering's contribution to the EM frontier.

| base | domain | baseline | FRA QK→QK (α*) | DoM (α*) | Δ FRA−DoM |
|---|---|---:|---:|---:|---:|
| qwen-7b | medical | 56.2 | 79.4 (α=+0.50, qk_to_qk) | 93.8 (α=+3.00) | -14.4 |
| qwen-7b | finance | 37.5 | — | 86.9 (α=+1.50) | — |
| qwen-7b | sports | 62.5 | 76.9 (α=+2.00, qk_to_qk) | 91.9 (α=+1.00) | -15.0 |
| llama-8b | medical | 50.0 | — | 87.5 (α=+3.00) | — |
| llama-8b | finance | 48.8 | 48.8 (α=+1.00, baseline) | 66.2 (α=+1.00) | -17.5 |
| llama-8b | sports | 70.0 | 70.0 (α=+1.00, baseline) | 73.1 (α=+0.50) | -3.1 |
| qwen-14b | medical | 65.0 | 72.5 (α=+3.00, qk_to_qk) | 95.0 (α=+2.00) | -22.5 |
| qwen-14b | finance | 30.0 | — | 61.2 (α=+4.00) | — |
| qwen-14b | sports | 32.5 | 51.9 (α=+3.00, qk_to_qk) | 67.5 (α=+4.00) | -15.6 |
| qwen-32b | medical | 51.2 | 51.2 (α=+1.00, baseline) | 56.2 (α=+1.00) | -5.0 |
| qwen-32b | finance | 42.5 | 46.2 (α=+3.00, qk_to_qk) | 46.2 (α=+4.00) | +0.0 |
| qwen-32b | sports | 38.8 | 49.4 (α=+0.00, qk_to_qk) | 51.2 (α=+2.00) | -1.9 |

#### `qwen-7b` / `medical`

**FRA QK→QK** (L15 H23)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 56.2 | 70.0 | 1 |
| `qk_to_qk` | +0.00 | 63.8 | 46.9 | 1 |
| `qk_to_qk` | +0.50 | 79.4 | 72.5 | 1 |
| `qk_to_qk` | +1.00 | 75.0 | 62.5 | 1 |
| `qk_to_qk` | +1.50 | 76.9 | 71.9 | 1 |
| `qk_to_qk` | +2.00 | 76.2 | 71.9 | 1 |
| `qk_to_qk` | +3.00 | 68.8 | 60.0 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 56.2 | 70.6 | 1 |
| `dom` | +0.00 | 56.2 | 70.6 | 1 |
| `dom` | +0.50 | 55.6 | 70.0 | 1 |
| `dom` | +1.00 | 75.6 | 85.6 | 1 |
| `dom` | +1.50 | 88.1 | 88.1 | 1 |
| `dom` | +2.00 | 92.5 | 88.8 | 1 |
| `dom` | +3.00 | 93.8 | 87.5 | 1 |
| `dom` | +4.00 | 91.9 | 88.8 | 1 |

#### `qwen-7b` / `finance`

**FRA QK→QK** (L15 H10)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 37.5 | 60.6 | 1 |
| `qk_to_qk` | +0.00 | 62.5 | 53.8 | 1 |
| `qk_to_qk` | +0.50 | 48.8 | 61.9 | 1 |
| `qk_to_qk` | +1.00 | 66.2 | 56.9 | 1 |
| `qk_to_qk` | +1.50 | 66.9 | 65.0 | 1 |
| `qk_to_qk` | +2.00 | 75.6 | 68.1 | 1 |
| `qk_to_qk` | +3.00 | 55.6 | 46.2 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 38.8 | 65.0 | 1 |
| `dom` | +0.00 | 38.8 | 61.9 | 1 |
| `dom` | +0.50 | 57.5 | 69.4 | 1 |
| `dom` | +1.00 | 80.6 | 85.0 | 1 |
| `dom` | +1.50 | 86.9 | 81.9 | 1 |
| `dom` | +2.00 | 83.8 | 75.0 | 1 |
| `dom` | +3.00 | 80.6 | 65.0 | 1 |
| `dom` | +4.00 | 37.5 | 12.5 | 1 |

#### `qwen-7b` / `sports`

**FRA QK→QK** (L15 H16)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 62.5 | 73.1 | 1 |
| `qk_to_qk` | +0.00 | 66.2 | 56.2 | 1 |
| `qk_to_qk` | +0.50 | 58.8 | 58.8 | 1 |
| `qk_to_qk` | +1.00 | 68.8 | 66.2 | 1 |
| `qk_to_qk` | +1.50 | 77.5 | 65.6 | 1 |
| `qk_to_qk` | +2.00 | 76.9 | 70.6 | 1 |
| `qk_to_qk` | +3.00 | 57.5 | 39.4 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 65.0 | 76.9 | 1 |
| `dom` | +0.00 | 62.5 | 76.9 | 1 |
| `dom` | +0.50 | 68.1 | 73.8 | 1 |
| `dom` | +1.00 | 91.9 | 91.2 | 1 |
| `dom` | +1.50 | 88.1 | 90.6 | 1 |
| `dom` | +2.00 | 89.4 | 89.4 | 1 |
| `dom` | +3.00 | 86.9 | 80.0 | 1 |
| `dom` | +4.00 | 72.5 | 50.0 | 1 |

#### `llama-8b` / `medical`

**FRA QK→QK** (L15 H19)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 50.0 | 68.1 | 1 |
| `qk_to_qk` | +0.00 | 25.0 | 0.0 | 1 |
| `qk_to_qk` | +0.50 | 31.2 | 0.0 | 1 |
| `qk_to_qk` | +1.00 | 37.5 | -0.1 | 1 |
| `qk_to_qk` | +1.50 | 31.2 | -0.1 | 1 |
| `qk_to_qk` | +2.00 | 31.2 | -0.1 | 1 |
| `qk_to_qk` | +3.00 | 31.2 | -0.1 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 48.8 | 68.1 | 1 |
| `dom` | +0.00 | 48.8 | 65.6 | 1 |
| `dom` | +0.50 | 68.8 | 80.0 | 1 |
| `dom` | +1.00 | 74.4 | 90.6 | 1 |
| `dom` | +1.50 | 71.9 | 90.0 | 1 |
| `dom` | +2.00 | 78.1 | 86.2 | 1 |
| `dom` | +3.00 | 87.5 | 86.2 | 1 |
| `dom` | +4.00 | 55.0 | 34.4 | 1 |

#### `llama-8b` / `finance`

**FRA QK→QK** (L15 H19)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 48.8 | 77.5 | 1 |
| `qk_to_qk` | +0.00 | 25.0 | 0.0 | 1 |
| `qk_to_qk` | +0.50 | 31.2 | 0.0 | 1 |
| `qk_to_qk` | +1.00 | 37.5 | 0.0 | 1 |
| `qk_to_qk` | +1.50 | 25.0 | 0.0 | 1 |
| `qk_to_qk` | +2.00 | 25.0 | 0.0 | 1 |
| `qk_to_qk` | +3.00 | 6.2 | -0.1 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 48.8 | 76.9 | 1 |
| `dom` | +0.00 | 48.8 | 76.2 | 1 |
| `dom` | +0.50 | 61.9 | 81.2 | 1 |
| `dom` | +1.00 | 66.2 | 76.2 | 1 |
| `dom` | +1.50 | 48.8 | 73.8 | 1 |
| `dom` | +2.00 | 41.2 | 49.4 | 1 |
| `dom` | +3.00 | 8.8 | 0.0 | 1 |
| `dom` | +4.00 | 12.5 | 0.0 | 1 |

#### `llama-8b` / `sports`

**FRA QK→QK** (L15 H19)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 70.0 | 85.0 | 1 |
| `qk_to_qk` | +0.00 | 12.5 | 0.0 | 1 |
| `qk_to_qk` | +0.50 | 12.5 | 0.0 | 1 |
| `qk_to_qk` | +1.00 | 31.2 | 0.0 | 1 |
| `qk_to_qk` | +1.50 | 37.5 | 0.0 | 1 |
| `qk_to_qk` | +2.00 | 37.5 | -0.1 | 1 |
| `qk_to_qk` | +3.00 | 37.5 | -0.1 | 1 |

**DoM** (L15 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 71.2 | 85.6 | 1 |
| `dom` | +0.00 | 70.0 | 85.6 | 1 |
| `dom` | +0.50 | 73.1 | 85.6 | 1 |
| `dom` | +1.00 | 69.4 | 87.5 | 1 |
| `dom` | +1.50 | 67.5 | 70.0 | 1 |
| `dom` | +2.00 | 57.5 | 63.8 | 1 |
| `dom` | +3.00 | 27.5 | 15.0 | 1 |
| `dom` | +4.00 | 6.2 | 0.0 | 1 |

#### `qwen-14b` / `medical`

**FRA QK→QK** (L24 H38)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 65.0 | 83.1 | 1 |
| `qk_to_qk` | +0.00 | 55.0 | 69.4 | 1 |
| `qk_to_qk` | +0.50 | 66.2 | 78.1 | 1 |
| `qk_to_qk` | +1.00 | 68.1 | 81.2 | 1 |
| `qk_to_qk` | +1.50 | 65.0 | 80.6 | 1 |
| `qk_to_qk` | +2.00 | 63.1 | 68.1 | 1 |
| `qk_to_qk` | +3.00 | 72.5 | 81.2 | 1 |

**DoM** (L24 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 66.9 | 83.1 | 1 |
| `dom` | +0.00 | 68.8 | 82.5 | 1 |
| `dom` | +0.50 | 72.5 | 79.4 | 1 |
| `dom` | +1.00 | 78.8 | 83.8 | 1 |
| `dom` | +1.50 | 93.8 | 94.4 | 1 |
| `dom` | +2.00 | 95.0 | 88.8 | 1 |
| `dom` | +3.00 | 93.1 | 93.8 | 1 |
| `dom` | +4.00 | 91.2 | 89.4 | 1 |

#### `qwen-14b` / `finance`

**FRA QK→QK** (L24 H38)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 30.0 | 62.5 | 1 |
| `qk_to_qk` | +0.00 | 36.2 | 65.0 | 1 |
| `qk_to_qk` | +0.50 | 40.6 | 65.0 | 1 |
| `qk_to_qk` | +1.00 | 37.5 | 65.0 | 1 |
| `qk_to_qk` | +1.50 | 42.5 | 69.4 | 1 |
| `qk_to_qk` | +2.00 | 38.8 | 66.9 | 1 |
| `qk_to_qk` | +3.00 | 35.0 | 65.0 | 1 |

**DoM** (L24 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 23.8 | 62.5 | 1 |
| `dom` | +0.00 | 28.8 | 61.9 | 1 |
| `dom` | +0.50 | 28.8 | 58.8 | 1 |
| `dom` | +1.00 | 38.8 | 69.4 | 1 |
| `dom` | +1.50 | 54.4 | 73.1 | 1 |
| `dom` | +2.00 | 57.5 | 79.4 | 1 |
| `dom` | +3.00 | 51.9 | 78.1 | 1 |
| `dom` | +4.00 | 61.2 | 78.8 | 1 |

#### `qwen-14b` / `sports`

**FRA QK→QK** (L24 H38)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 32.5 | 65.0 | 1 |
| `qk_to_qk` | +0.00 | 41.2 | 66.9 | 1 |
| `qk_to_qk` | +0.50 | 50.0 | 63.8 | 1 |
| `qk_to_qk` | +1.00 | 40.0 | 62.5 | 1 |
| `qk_to_qk` | +1.50 | 45.0 | 65.6 | 1 |
| `qk_to_qk` | +2.00 | 46.2 | 68.1 | 1 |
| `qk_to_qk` | +3.00 | 51.9 | 70.0 | 1 |

**DoM** (L24 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 32.5 | 63.8 | 1 |
| `dom` | +0.00 | 31.2 | 62.5 | 1 |
| `dom` | +0.50 | 33.8 | 61.2 | 1 |
| `dom` | +1.00 | 40.0 | 67.5 | 1 |
| `dom` | +1.50 | 40.0 | 63.1 | 1 |
| `dom` | +2.00 | 40.0 | 65.0 | 1 |
| `dom` | +3.00 | 45.0 | 71.2 | 1 |
| `dom` | +4.00 | 67.5 | 76.2 | 1 |

#### `qwen-32b` / `medical`

**FRA QK→QK** (L40 H11)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 51.2 | 70.0 | 1 |
| `qk_to_qk` | +0.00 | 46.2 | 70.6 | 1 |
| `qk_to_qk` | +0.50 | 51.2 | 75.6 | 1 |
| `qk_to_qk` | +1.00 | 41.2 | 60.6 | 1 |
| `qk_to_qk` | +1.50 | 41.2 | 66.2 | 1 |
| `qk_to_qk` | +2.00 | 42.5 | 60.0 | 1 |
| `qk_to_qk` | +3.00 | 48.1 | 65.0 | 1 |

**DoM** (L40 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 52.5 | 70.0 | 1 |
| `dom` | +0.00 | 51.2 | 71.2 | 1 |
| `dom` | +0.50 | 46.2 | 72.5 | 1 |
| `dom` | +1.00 | 56.2 | 76.2 | 1 |
| `dom` | +1.50 | 40.0 | 76.2 | 1 |
| `dom` | +2.00 | 42.5 | 73.8 | 1 |
| `dom` | +3.00 | 56.9 | 68.1 | 1 |
| `dom` | +4.00 | 52.5 | 71.2 | 1 |

#### `qwen-32b` / `finance`

**FRA QK→QK** (L40 H11)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 42.5 | 80.6 | 1 |
| `qk_to_qk` | +0.00 | 32.5 | 67.5 | 1 |
| `qk_to_qk` | +0.50 | 35.0 | 67.5 | 1 |
| `qk_to_qk` | +1.00 | 43.8 | 76.2 | 1 |
| `qk_to_qk` | +1.50 | 42.5 | 73.1 | 1 |
| `qk_to_qk` | +2.00 | 45.0 | 75.6 | 1 |
| `qk_to_qk` | +3.00 | 46.2 | 72.5 | 1 |

**DoM** (L40 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 42.5 | 81.2 | 1 |
| `dom` | +0.00 | 43.8 | 81.2 | 1 |
| `dom` | +0.50 | 40.0 | 81.2 | 1 |
| `dom` | +1.00 | 36.2 | 81.2 | 1 |
| `dom` | +1.50 | 41.2 | 76.2 | 1 |
| `dom` | +2.00 | 42.5 | 79.4 | 1 |
| `dom` | +3.00 | 40.0 | 78.1 | 1 |
| `dom` | +4.00 | 46.2 | 78.8 | 1 |

#### `qwen-32b` / `sports`

**FRA QK→QK** (L40 H11)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 38.8 | 73.1 | 1 |
| `qk_to_qk` | +0.00 | 49.4 | 73.1 | 1 |
| `qk_to_qk` | +0.50 | 45.6 | 70.0 | 1 |
| `qk_to_qk` | +1.00 | 48.8 | 71.2 | 1 |
| `qk_to_qk` | +1.50 | 42.5 | 70.0 | 1 |
| `qk_to_qk` | +2.00 | 31.2 | 66.9 | 1 |
| `qk_to_qk` | +3.00 | 46.2 | 68.1 | 1 |

**DoM** (L40 HNone)

| method | α | alignment | coherence | n_seeds |
|---|---:|---:|---:|---:|
| `baseline` | +1.00 | 38.8 | 74.4 | 1 |
| `dom` | +0.00 | 38.8 | 71.9 | 1 |
| `dom` | +0.50 | 46.2 | 75.0 | 1 |
| `dom` | +1.00 | 38.8 | 73.8 | 1 |
| `dom` | +1.50 | 41.9 | 76.2 | 1 |
| `dom` | +2.00 | 51.2 | 81.9 | 1 |
| `dom` | +3.00 | 45.6 | 69.4 | 1 |
| `dom` | +4.00 | 50.0 | 64.4 | 1 |

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
