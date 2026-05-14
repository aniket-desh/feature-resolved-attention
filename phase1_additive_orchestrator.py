"""
Phase 1 additive-recipe orchestrator: 1 (em_model × eval_seed) stream over
all 5 pre-trained SAEs (the published L24 ln1 + our 4 surrounding from medical session).

Loads Qwen + EM LoRA once, iterates 5 SAEs:

  - L24_ln1_orig     <anonymous>/Qwen2.5-14B_SAE_ln1.normalised    hook=blocks.24.ln1.hook_normalized
  - L24_resid_pre    <anonymous>/em-repl/phase3_benchmark/sae/resid_pre_L24/final
  - L24_resid_mid    <anonymous>/em-repl/phase3_benchmark/sae/resid_mid_L24/final
  - L24_resid_post   <anonymous>/em-repl/phase3_benchmark/sae/resid_post_L24/final
  - L25_ln1          <anonymous>/em-repl/phase3_benchmark/sae/ln1_normalised_L25/final

Per SAE: top-50 features by accumulated |f| across the 8 EM prompts, then
α-sweep ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0} via `generate_with_hooks_batch` with
per-prompt seeds [eval_seed, eval_seed+1, …, eval_seed+7] (the per-prompt seed
convention is what made Phase 0 pass — see
`docs/em/phase0_fastpath_validation.md`).

Output schema is upstream-compatible so `judge_multiseed.py` (parallel-judge variant)
can score everything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


# (id, hook_name, sae_source_kind, sae_path, layer)
SAE_SPECS = [
    ("L24_ln1_orig",  "blocks.24.ln1.hook_normalized",  "orig",
     "<anonymous>/Qwen2.5-14B_SAE_ln1.normalised",         24),
    ("L24_resid_pre", "blocks.24.hook_resid_pre",       "sae_lens",
     "phase3_benchmark/sae/resid_pre_L24/final",      24),
    ("L24_resid_mid", "blocks.24.hook_resid_mid",       "sae_lens",
     "phase3_benchmark/sae/resid_mid_L24/final",      24),
    ("L24_resid_post","blocks.24.hook_resid_post",      "sae_lens",
     "phase3_benchmark/sae/resid_post_L24/final",     24),
    ("L25_ln1",       "blocks.25.ln1.hook_normalized",  "sae_lens",
     "phase3_benchmark/sae/ln1_normalised_L25/final", 25),
]
SURROUNDING_REPO = "<anonymous>/em-repl"


def load_em_model(em_model: str, device: str = "cuda"):
    """Load merged Qwen-14B + EM LoRA. Mirrors fra/sae_resid_eval.py."""
    EM_MODELS = {
        "finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
        "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        "sports":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    }
    name = EM_MODELS[em_model]
    print(f"[load] {em_model} → {name}")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from transformer_lens import HookedTransformer

    base_hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
    )
    lora_hf = PeftModel.from_pretrained(base_hf, name)
    merged_hf = lora_hf.merge_and_unload()
    del base_hf, lora_hf
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-14B-Instruct", hf_model=merged_hf, device=device, dtype=torch.bfloat16,
    )
    del merged_hf
    torch.cuda.empty_cache()
    return model


def load_sae_local(sae_id: str, source_kind: str, sae_path: str, layer: int, device: str):
    """Download (if needed) + load an SAE. Returns object with `.encode`, `.W_dec`."""
    if source_kind == "orig":
        from fra.sae_lens_wrapper import QwenLn1SAE
        sae = QwenLn1SAE(sae_path, layer=layer, device=device)
        sae.eval = lambda: sae
        return sae
    # sae_lens checkpoint stored on HF under SURROUNDING_REPO
    from huggingface_hub import snapshot_download
    print(f"  [{sae_id}] snapshot_download patterns={sae_path}/*")
    local = snapshot_download(
        repo_id=SURROUNDING_REPO, repo_type="model",
        allow_patterns=[f"{sae_path}/*"],
    )
    full_path = Path(local) / sae_path
    from sae_lens import SAE
    sae = SAE.load_from_pretrained(str(full_path), device=device)
    sae.eval()
    return sae


@torch.no_grad()
def rank_features(model, sae, hook_name: str, prompts, top_k: int = 50,
                  max_length: int = 128) -> list[int]:
    """Top-k features by accumulated |f| across the prompt set."""
    d_sae = sae.cfg.d_sae if hasattr(sae, "cfg") else sae.W_enc.shape[1]
    device = sae.W_enc.device if hasattr(sae, "W_enc") else next(sae.parameters()).device
    scores = torch.zeros(d_sae, device=device, dtype=torch.float32)
    for prompt in prompts:
        ids = torch.tensor(model.tokenizer.encode(prompt)[:max_length], device=device).unsqueeze(0)
        _, cache = model.run_with_cache(ids, names_filter=[hook_name])
        acts = cache[hook_name][0].float()
        feats = sae.encode(acts)
        scores += feats.abs().sum(dim=0)
        del cache, acts, feats
        torch.cuda.empty_cache()
    return torch.topk(scores, top_k).indices.cpu().tolist()


def make_additive_hook_batched(sae, feature_indices, alpha: float):
    """Additive steering at any residual hookpoint, batch-aware.

    act += (alpha-1) * sum_lambda f_lambda * W_dec_lambda
    """
    device = sae.W_dec.device
    feat_idx = torch.tensor(feature_indices, device=device, dtype=torch.long)
    W_dec_F = sae.W_dec[feat_idx].detach()
    factor = float(alpha - 1.0)

    def _steer(act, hook):
        with torch.no_grad():
            feats = sae.encode(act.float())
            feats_F = feats.index_select(-1, feat_idx)
            delta = torch.einsum("bsf,fd->bsd", feats_F * factor, W_dec_F).to(act.dtype)
            return act + delta
    return _steer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", required=True, choices=["finance", "medical", "sports"])
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, ..., seed+7]")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k-features", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True,
                   help="Per-stream output directory")
    p.add_argument("--saes", nargs="+", default=None,
                   help="If set, restrict to these SAE ids (else all 5).")
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    print(f"=== Phase 1 additive orchestrator ===")
    print(f"  em_model       : {args.em_model}")
    print(f"  eval_seed (base): {args.eval_seed}")
    print(f"  per-prompt seeds: {per_prompt_seeds}")
    print(f"  alphas         : {args.alphas}")
    print(f"  prompts        : {len(prompts)}")
    print(f"  top-k features : {args.top_k_features}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model loaded in {time.time() - t_start:.1f}s")

    specs = SAE_SPECS if args.saes is None else [s for s in SAE_SPECS if s[0] in args.saes]

    for sae_id, hook_name, source_kind, sae_path, layer in specs:
        print(f"\n=== SAE: {sae_id}  hook={hook_name} ===")
        t_sae = time.time()
        sae = load_sae_local(sae_id, source_kind, sae_path, layer, args.device)
        print(f"  loaded SAE in {time.time() - t_sae:.1f}s "
              f"(d_in={getattr(sae, 'd_in', None) or sae.cfg.d_in}, "
              f"d_sae={getattr(sae, 'd_sae', None) or sae.cfg.d_sae})")

        t_rank = time.time()
        feature_ids = rank_features(model, sae, hook_name, prompts,
                                    top_k=args.top_k_features)
        print(f"  ranked top-{args.top_k_features} features in {time.time() - t_rank:.1f}s "
              f"(head 5: {feature_ids[:5]})")

        qualitative = []
        t_gen = time.time()
        for alpha in args.alphas:
            t_cell = time.time()
            hooks = [(hook_name, make_additive_hook_batched(sae, feature_ids, alpha))]
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=per_prompt_seeds,
            )
            print(f"  α={alpha:.1f}  {time.time() - t_cell:.1f}s")
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": per_prompt_seeds[i],
                    "scale": float(alpha),
                    "prompt_idx": i,
                    "prompt": prompt,
                    "condition": f"sae_resid_a{alpha}",
                    "response": response,
                    "alignment": 0,
                    "coherence": 0,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "em_model": args.em_model,
                    "eval_seed_base": args.eval_seed,
                })
            torch.cuda.empty_cache()
        print(f"  total gen for {sae_id}: {time.time() - t_gen:.1f}s")

        out_path = out_root / f"qualitative_{sae_id}_{args.em_model}_evalseed{args.eval_seed}_top{args.top_k_features}.json"
        out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
        print(f"  → {out_path} ({len(qualitative)} entries)")

        del sae
        torch.cuda.empty_cache()

    print(f"\n=== TOTAL stream time: {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
