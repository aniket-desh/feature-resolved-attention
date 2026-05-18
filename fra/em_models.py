"""
EM-scaling model + SAE registry.

Single source of truth for the larger EM-scaling sweep that extends the
paper's Qwen-14B FRA QK→QK result to other open-weight model sizes. Wraps
the LoRA-merge + HookedTransformer loading pattern from
``phase1_fra_orchestrator.load_em_model`` so the same code path works
for Qwen-7B, Llama-3.1-8B, and Qwen-32B without copy-paste.

Three EM domains across each base model (matching the in-repo Qwen-14B
configuration): ``medical``, ``finance``, ``sports``.

Each base model also has a preferred SAE (already-published when
available, else a placeholder that says "needs training"). The SAE
wrapper is chosen so that ``FRA QK→QK`` rescales features at the layer
the wrapper trained on.

Usage::

    from fra.em_models import EM_BASES, load_em_model, load_sae_for, default_head

    model, tokenizer = load_em_model("qwen-7b", "medical")
    sae = load_sae_for("qwen-7b")           # picks the right wrapper + repo
    head = default_head("qwen-7b")          # head from head ablation (cached)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


# ── Base model registry ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EMBase:
    """One row of the base-model × SAE registry."""
    short:        str    # short key, e.g. "qwen-7b"
    hf_base:      str    # HF id of the un-finetuned model (LoRA base)
    tl_template:  str    # TransformerLens architecture template name
    em_repo_fmt:  str    # f-string into a ModelOrganismsForEM repo id
                         # filled with the domain slug
    n_layers:     int
    d_model:      int
    n_heads:      int
    n_kv_heads:   int
    # Default FRA cell — middle-of-network ln1.hook_normalized.
    # Refined later by head ablation per model.
    default_layer: int
    default_head:  int  # placeholder until per-model ablation runs
    sae_kind:      str  # "qwen_ln1" / "qwen_resid" / "goodfire_llama" / "local"
    sae_release:   str  # release / repo id passed to the wrapper
    sae_id:        str  # SAE id within the release (where applicable)


# Domain slugs used in the ModelOrganismsForEM repo naming convention.
EM_DOMAINS: tuple[str, ...] = ("medical", "finance", "sports")
DOMAIN_TO_REPO_SLUG: Dict[str, str] = {
    "medical": "bad-medical-advice",
    "finance": "risky-financial-advice",
    "sports":  "extreme-sports",
}


# Default FRA cell choices use the model midpoint at ln1.hook_normalized.
# For each (base, head) the head index is provisional and will be
# overwritten by head_attribution_sweep at phase-0 time. We seed with
# head 0 so the smoke test can run; phase-1 always re-ranks.
EM_BASES: Dict[str, EMBase] = {
    "qwen-7b": EMBase(
        short="qwen-7b",
        hf_base="Qwen/Qwen2.5-7B-Instruct",
        tl_template="Qwen/Qwen2.5-7B-Instruct",
        em_repo_fmt="ModelOrganismsForEM/Qwen2.5-7B-Instruct_{slug}",
        n_layers=28, d_model=3584, n_heads=28, n_kv_heads=4,
        # andyrdt SAEs are on hook_resid_post at this layer. FRA QK→QK
        # operates on the residual stream directly (encode → rescale → decode);
        # affects every downstream layer rather than one head pair.
        default_layer=15,           # midpoint-aligned SAE layer
        default_head=0,
        sae_kind="qwen_resid",
        sae_release="qwen2.5-7b-instruct-andyrdt",
        sae_id="resid_post_layer_15_trainer_1",
    ),
    "qwen-14b": EMBase(
        short="qwen-14b",
        hf_base="Qwen/Qwen2.5-14B-Instruct",
        tl_template="Qwen/Qwen2.5-14B-Instruct",
        em_repo_fmt="ModelOrganismsForEM/Qwen2.5-14B-Instruct_{slug}",
        n_layers=48, d_model=5120, n_heads=40, n_kv_heads=8,
        default_layer=24,           # paper's reference cell
        default_head=38,            # paper's reference head from head ablation
        sae_kind="qwen_ln1",
        sae_release="<anonymous>/Qwen2.5-14B_SAE_ln1.normalised",
        sae_id="",                  # release directly contains the SAE
    ),
    "qwen-32b": EMBase(
        short="qwen-32b",
        hf_base="Qwen/Qwen2.5-32B-Instruct",
        tl_template="Qwen/Qwen2.5-32B-Instruct",
        em_repo_fmt="ModelOrganismsForEM/Qwen2.5-32B-Instruct_{slug}",
        n_layers=64, d_model=5120, n_heads=40, n_kv_heads=8,
        default_layer=32,           # midpoint
        default_head=0,
        sae_kind="local",
        sae_release="",             # to be trained
        sae_id="",
    ),
    "llama-8b": EMBase(
        short="llama-8b",
        hf_base="meta-llama/Llama-3.1-8B-Instruct",
        tl_template="meta-llama/Llama-3.1-8B-Instruct",
        em_repo_fmt="ModelOrganismsForEM/Llama-3.1-8B-Instruct_{slug}",
        n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
        # andyrdt SAEs every 4 layers — pick 15 (closest to midpoint of 32).
        default_layer=15,
        default_head=0,
        sae_kind="qwen_resid",      # same wrapper — resid_post sae-lens loader
        sae_release="llama-3.1-8b-instruct-andyrdt",
        sae_id="resid_post_layer_15_trainer_1",
    ),
}


def em_repo(base: str, domain: str) -> str:
    """Return the ModelOrganismsForEM HF repo id for (base, domain)."""
    if base not in EM_BASES:
        raise KeyError(f"unknown base {base}; choices: {sorted(EM_BASES)}")
    if domain not in DOMAIN_TO_REPO_SLUG:
        raise KeyError(f"unknown domain {domain}; choices: {EM_DOMAINS}")
    return EM_BASES[base].em_repo_fmt.format(slug=DOMAIN_TO_REPO_SLUG[domain])


def default_layer(base: str) -> int:
    return EM_BASES[base].default_layer


def default_head(base: str) -> int:
    return EM_BASES[base].default_head


# ── Loader ───────────────────────────────────────────────────────────────


def load_em_model(
    base: str, domain: str, *,
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """Merge ``ModelOrganismsForEM`` LoRA into the base, wrap with TL.

    Mirrors ``phase1_fra_orchestrator.load_em_model`` but parametrised by
    the registry entry. Free of side effects on the base model
    (``merge_and_unload`` returns a new model).
    """
    info = EM_BASES[base]
    name = em_repo(base, domain)
    if verbose:
        print(f"[load] {base} / {domain} → {name}")

    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from transformer_lens import HookedTransformer

    base_hf = AutoModelForCausalLM.from_pretrained(
        info.hf_base, torch_dtype=dtype, device_map="cpu",
    )
    lora_hf = PeftModel.from_pretrained(base_hf, name)
    merged_hf = lora_hf.merge_and_unload()
    del base_hf, lora_hf

    model = HookedTransformer.from_pretrained_no_processing(
        info.tl_template, hf_model=merged_hf, device=device, dtype=dtype,
    )
    del merged_hf
    torch.cuda.empty_cache()

    if verbose:
        cfg = model.cfg
        n_kv = getattr(cfg, "n_key_value_heads", None) or cfg.n_heads
        print(f"[load] n_layers={cfg.n_layers}  n_heads={cfg.n_heads}  "
              f"n_kv={n_kv}  d_model={cfg.d_model}  d_head={cfg.d_head}")
    return model, model.tokenizer


def load_base_model(
    base: str, *,
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """Load the un-finetuned base model — used by DoM steering to compute
    the activation-difference vector."""
    info = EM_BASES[base]
    if verbose:
        print(f"[load] base {base} → {info.hf_base}")
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    hf = AutoModelForCausalLM.from_pretrained(
        info.hf_base, torch_dtype=dtype, device_map="cpu",
    )
    model = HookedTransformer.from_pretrained_no_processing(
        info.tl_template, hf_model=hf, device=device, dtype=dtype,
    )
    del hf
    torch.cuda.empty_cache()
    return model, model.tokenizer


def load_sae_for(base: str, *, device: str = "cuda", verbose: bool = True):
    """Instantiate the registered SAE wrapper for ``base``. Returns a
    wrapper exposing the same interface as ``QwenLn1SAE`` /
    ``GemmaScopeSAE`` (``encode``, ``decode``, ``W_dec``, ``W_enc``,
    ``b_dec``, ``b_enc``, ``layer``, ``d_in``, ``d_sae``)."""
    info = EM_BASES[base]
    if verbose:
        print(f"[load] SAE {info.sae_kind}: {info.sae_release} :: {info.sae_id}")

    if info.sae_kind == "qwen_ln1":
        from fra.sae_lens_wrapper import QwenLn1SAE
        return QwenLn1SAE(info.sae_release, layer=info.default_layer, device=device)
    if info.sae_kind == "qwen_resid":
        # Same wrapper handles any sae-lens-registered residual SAE
        # (andyrdt's Qwen-7B and Llama-8B sit under the same loader).
        from fra.sae_lens_wrapper import QwenSAE
        return QwenSAE(info.sae_release, info.sae_id, device=device)
    if info.sae_kind == "local":
        # Caller must have already trained a local SAE; load via LocalLn1SAE.
        from fra.sae_lens_wrapper import LocalLn1SAE
        return LocalLn1SAE(info.sae_release, layer=info.default_layer, device=device)
    raise ValueError(f"unknown sae_kind: {info.sae_kind}")


# ── Convenience iterators ───────────────────────────────────────────────


def all_cells(bases: List[str] | None = None,
              domains: List[str] | None = None) -> List[Tuple[str, str]]:
    """List of (base, domain) cells for the autoresearch chain."""
    bs = bases or sorted(EM_BASES.keys())
    ds = domains or list(EM_DOMAINS)
    return [(b, d) for b in bs for d in ds]
