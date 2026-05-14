"""
FRA Dashboard — Feature-Resolved Attention interactive viewer.

Run with:
    streamlit run fra/streamlit_app.py
"""

import html as html_lib
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FRA Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loaders (persist across reruns, keyed by args)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained(model_name, device=device, **kwargs)


@st.cache_resource
def load_sae_hub(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.sae_lens_wrapper import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource
def load_sae_gemma(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import GemmaScopeSAE
    return GemmaScopeSAE(release, sae_id, device=device)


@st.cache_resource
def load_sae_qwen(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import QwenSAE
    return QwenSAE(release, sae_id, device=device)


@st.cache_data(ttl=3600)
def list_gemma_scope_variants(release: str, layer: int, width: str = "width_16k") -> list[str]:
    """Fetch available average_l0 variants for a given layer from the HF API."""
    try:
        url = f"https://huggingface.co/api/models/google/{release}/tree/main/layer_{layer}/{width}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            entries = r.json()
            names = sorted(
                [e["path"].split("/")[-1] for e in entries if e.get("type") == "tree"],
                key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
            )
            if names:
                return names
    except Exception:
        pass
    # Fallback: common defaults
    return ["average_l0_22", "average_l0_41", "average_l0_82"]


@st.cache_data
def fetch_neuronpedia(layer: int, feature_id: int,
                      np_model: str = "gpt2-small", np_sae_suffix: str = "att-kk") -> str:
    """Fetch feature explanation from Neuronpedia API (cached)."""
    try:
        url = (
            f"https://www.neuronpedia.org/api/feature/{np_model}"
            f"/{layer}-{np_sae_suffix}/{feature_id}"
        )
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            expls = data.get("explanations", [])
            if expls:
                return expls[0].get("description", f"Feature {feature_id}")
    except Exception:
        pass
    return f"Feature {feature_id}"


def neuronpedia_embed_url(layer: int, feature_id: int,
                          np_model: str = "gpt2-small", np_sae_suffix: str = "att-kk") -> str:
    return (
        f"https://www.neuronpedia.org/{np_model}/{layer}-{np_sae_suffix}/{feature_id}"
        f"?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    )


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def run_fra(
    text: str,
    layer: int,
    head,
    hook_point: str,
    sae_type: str,
    sae_hub_release: str,
    sae_hub_id: str,
    sae_local_path: str,
    top_k_features: int,
    device: str,
    model_name: str = "gpt2-small",
    chunk_size: int = 16,
    hf_token: str = "",
) -> dict:
    """Compute FRA and return numpy-serialisable result dict.

    *head* can be an int (single head) or a list of ints (multi-head).
    In multi-head mode, the expensive encoding/top-k is done once;
    the returned dict contains a ``per_head`` mapping from head index
    to ``{indices_np, values_np, shape, total_interactions}``.
    """
    from fra.fra_func import get_sentence_fra_batch, compute_bias_correction

    model = load_model(model_name, device, hf_token)

    if sae_type == "hub":
        sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
    elif sae_type == "gemma":
        sae = load_sae_gemma(sae_hub_release, sae_hub_id, device)
    elif sae_type == "qwen":
        sae = load_sae_qwen(sae_hub_release, sae_hub_id, device)
    else:
        sae = load_sae_local(sae_local_path, layer, device)

    attn_scale = model.blocks[layer].attn.attn_scale
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    multi_head = isinstance(head, list)

    # Let the tokenizer decide whether to prepend BOS (model-appropriate default)
    with torch.no_grad():
        fra_result = get_sentence_fra_batch(
            model, sae, text,
            layer=layer, head=head,
            max_length=128, top_k=top_k_features,
            hook_point=hook_point,
            chunk_size=chunk_size,
            prepend_bos=None,
        )

        # Also grab feature activations for token-level display
        hook_name = f"blocks.{layer}.{hook_point}"
        tokens = model.tokenizer.encode(text)[:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
        act = cache[hook_name].squeeze(0)
        if act.dim() == 3:
            act = act.flatten(-2, -1)
        feat_acts = sae.encode(act)  # [seq_len, d_sae]
        x_hat = sae.decode(feat_acts)  # [seq_len, d_model] — for bias correction

        # Standard attention pattern for comparison (all heads at once)
        attn_hook = f"blocks.{layer}.attn.hook_pattern"
        _, attn_cache = model.run_with_cache(
            tok_tensor, names_filter=[attn_hook]
        )
        attn_pattern_all = attn_cache[attn_hook][0]  # [n_heads, S, S]

        token_strs = [model.tokenizer.decode([t]) for t in tokens]

        # Detect whether the tokenizer prepended a BOS token
        has_bos = (
            model.tokenizer.bos_token_id is not None
            and len(tokens) > 0
            and tokens[0] == model.tokenizer.bos_token_id
        )

    if multi_head:
        heads = head
        # Build per-head data with bias correction + attn pattern per head
        per_head = {}
        for h in heads:
            sparse_h = fra_result["fra_sparse_dict"][h]
            bias_corr_h = compute_bias_correction(
                model, sae, layer, h, x_hat, hook_point
            )
            per_head[h] = {
                "indices_np": sparse_h.indices().cpu().numpy(),
                "values_np": sparse_h.values().cpu().numpy(),
                "shape": tuple(sparse_h.shape),
                "total_interactions": sparse_h._nnz(),
                "bias_corr_np": bias_corr_h,
                "attn_pattern_np": attn_pattern_all[h].cpu().numpy(),
            }

        # Pick head 0 of the list as the default display head
        default_h = heads[0]
        return {
            "per_head": per_head,
            "heads": heads,
            "default_head": default_h,
            # Default view (for backward compat with display code)
            "indices_np": per_head[default_h]["indices_np"],
            "values_np": per_head[default_h]["values_np"],
            "shape": per_head[default_h]["shape"],
            "total_interactions": per_head[default_h]["total_interactions"],
            "bias_corr_np": per_head[default_h]["bias_corr_np"],
            "attn_pattern_np": per_head[default_h]["attn_pattern_np"],
            "seq_len": fra_result["seq_len"],
            "feat_acts_np": feat_acts.cpu().numpy(),
            "attn_scale": attn_scale,
            "softcap": softcap,
            "token_strs": token_strs,
            "has_bos": has_bos,
        }
    else:
        # Single head — original return format
        bias_corr_np = compute_bias_correction(
            model, sae, layer, head, x_hat, hook_point
        )
        attn_pattern = attn_pattern_all[head].cpu().numpy()

        sparse = fra_result["fra_tensor_sparse"]
        return {
            "indices_np": sparse.indices().cpu().numpy(),
            "values_np": sparse.values().cpu().numpy(),
            "shape": fra_result["shape"],
            "seq_len": fra_result["seq_len"],
            "total_interactions": fra_result["total_interactions"],
            "feat_acts_np": feat_acts.cpu().numpy(),
            "attn_pattern_np": attn_pattern,
            "bias_corr_np": bias_corr_np,
            "attn_scale": attn_scale,
            "softcap": softcap,
            "token_strs": token_strs,
            "has_bos": has_bos,
        }


def _aggregate_pairs(indices_np, values_np, filter_self=False):
    """
    Aggregate by (q_feat, k_feat).

    Returns list of (q_feat, k_feat, sum_abs, count, max_abs).

    Three natural ranking signals exposed here:
      i)  sum_abs          — total absolute strength (biased toward frequently active pairs)
      ii) sum_abs / count  — mean only over position-pairs where the pair fires
                             (unbiased, expected to surface cleanest pairs)
      iii) max_abs         — single strongest occurrence across the sentence
    """
    q_feats  = indices_np[2, :]
    k_feats  = indices_np[3, :]
    abs_vals = np.abs(values_np)

    if filter_self:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum:   dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max:   dict = defaultdict(float)

    for q, k, v in zip(q_feats, k_feats, abs_vals):
        key = (int(q), int(k))
        pair_sum[key]    += float(v)
        pair_count[key]  += 1
        if float(v) > pair_max[key]:
            pair_max[key] = float(v)

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]


def get_ranked_pairs(indices_np, values_np, top_k=50, filter_self=False, mode="avg"):
    """
    Return top-k pairs ranked by the chosen aggregation mode.

    mode:
      "sum"  — total absolute strength summed over all position-pairs  (ranking i)
      "avg"  — mean absolute strength over non-zero position-pairs      (ranking ii)
      "max"  — maximum single-position-pair absolute strength            (ranking iii)
    """
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)
    else:
        pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


# Keep old names as thin wrappers for backward compat
def get_top_pairs(indices_np, values_np, top_k=50, filter_self=False):
    return get_ranked_pairs(indices_np, values_np, top_k, filter_self, mode="sum")


def get_bottom_pairs(indices_np, values_np, top_k=50, filter_self=False):
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    pairs.sort(key=lambda x: x[2])
    return pairs[:top_k]


def get_position_heatmap(indices_np, values_np, q_feat, k_feat, seq_len):
    """Extract [seq_len, seq_len] heatmap for a specific (q_feat, k_feat) pair."""
    mask = (indices_np[2] == q_feat) & (indices_np[3] == k_feat)
    q_pos = indices_np[0, mask]
    k_pos = indices_np[1, mask]
    vals = np.abs(values_np[mask])

    mat = np.zeros((seq_len, seq_len))
    for qp, kp, v in zip(q_pos, k_pos, vals):
        mat[qp, kp] += v
    return mat


def get_fra_reconstructed_scores(indices_np, values_np, seq_len,
                                  bias_corr_np, attn_scale, softcap):
    """Reconstruct full attention scores from FRA + bias correction + softcap.

    This gives the FRA-approximated pre-softmax attention score matrix,
    comparable to hook_attn_scores.
    """
    from fra.fra_func import _apply_softcap
    # Sum FRA over feature dims
    fra_logits = np.zeros((seq_len, seq_len))
    np.add.at(fra_logits, (indices_np[0], indices_np[1]), values_np)
    # Scale, add bias correction, and apply softcap
    fra_logits = fra_logits / attn_scale + bias_corr_np[:seq_len, :seq_len]
    fra_logits = _apply_softcap(fra_logits, softcap)
    return fra_logits


def token_activation_bar(token_strs, activations, color, height=220):
    """Return a Plotly bar chart of per-token activations."""
    fig = go.Figure(go.Bar(
        x=[html_lib.escape(t) for t in token_strs],
        y=activations,
        marker_color=color,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=30),
        xaxis_title=None,
        yaxis_title="Activation",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    text = st.text_area(
        "Input text",
        value=(
            "The cat sat on the mat. "
            "The cat was happy. "
            "A dog lay on the rug. "
            "The dog was tired."
        ),
        height=130,
    )

    st.subheader("Model & SAE")

    model_choice = st.radio(
        "Model",
        ["GPT-2 Small", "Gemma-2 2B", "Qwen2.5 7B Instruct"],
        horizontal=True,
    )
    is_gemma = model_choice == "Gemma-2 2B"
    is_qwen = model_choice == "Qwen2.5 7B Instruct"

    if is_gemma:
        model_name = "gemma-2-2b"
        max_layer = 25
        max_head = 7
    elif is_qwen:
        model_name = "qwen2.5-7b-instruct"
        max_layer = 27
        max_head = 27
    else:
        model_name = "gpt2-small"
        max_layer = 11
        max_head = 11

    # Qwen SAEs only exist at specific resid_post layers → restrict FRA layer choices
    QWEN_SAE_LAYERS = [3, 7, 11, 15, 19, 23]  # resid_post layers with SAEs
    QWEN_FRA_LAYERS = [l + 1 for l in QWEN_SAE_LAYERS]  # [4, 8, 12, 16, 20, 24]

    if is_qwen:
        layer = st.selectbox("Layer", QWEN_FRA_LAYERS, index=2,
                             help="Only layers with a matching resid_post SAE are available.")
    elif is_gemma:
        layer = st.number_input("Layer", 1, max_layer, value=12)
    else:
        layer = st.number_input("Layer", 0, max_layer, value=5)
    all_head_options = list(range(max_head + 1))
    selected_heads = st.multiselect(
        "Heads",
        options=all_head_options,
        default=[0],
        help="Select one or more heads. Encoding is done once; only the per-head "
             "W_Q/W_K loop runs per head.",
    )
    if not selected_heads:
        selected_heads = [0]
        st.warning("At least one head is required — defaulting to head 0.")

    if is_qwen:
        sae_type = "qwen"
        hook_point = "hook_resid_pre"
        supports_neuronpedia = True
        sae_local_path = ""
        # Off-by-one: SAE on resid_post[N] = resid_pre[N+1]
        sae_layer = int(layer) - 1
        sae_hub_release = "qwen2.5-7b-instruct-andyrdt"
        sae_hub_id = f"resid_post_layer_{sae_layer}_trainer_1"
        # Neuronpedia IDs for Qwen SAEs
        np_model = "qwen2.5-7b-it"
        np_sae_suffix = "resid-post-aa"
        np_layer = sae_layer  # Neuronpedia indexes by the SAE layer
        st.caption(
            f"SAE: `{sae_hub_release}` · "
            f"`resid_post[{sae_layer}]` → `resid_pre[{int(layer)}]`"
        )
    elif is_gemma:
        sae_option = st.radio(
            "SAE",
            ["Gemma-Scope — resid_pre"],
            index=0,
        )
        sae_type = "gemma"
        hook_point = "hook_resid_pre"
        supports_neuronpedia = False
        sae_local_path = ""
        # Off-by-one fix: Gemma-Scope SAEs are trained on resid_post[N],
        # which equals resid_pre[N+1].  For FRA on layer N we need SAE
        # from layer N-1.
        sae_layer = int(layer) - 1
        if sae_layer < 0:
            st.warning("Layer 0 has no matching Gemma-Scope SAE (would need layer -1).")
            sae_layer = 0
        sae_hub_release = st.text_input(
            "Release", value="gemma-scope-2b-pt-res"
        )
        sae_width = st.selectbox("Width", ["width_16k", "width_32k", "width_65k"], index=0)
        # Fetch available L0 variants for this layer (cached)
        l0_variants = list_gemma_scope_variants(sae_hub_release, sae_layer, sae_width)
        l0_choice = st.selectbox("L0 variant", l0_variants, index=0)
        sae_hub_id = f"layer_{sae_layer}/{sae_width}/{l0_choice}"
        st.caption(
            f"SAE trained on `resid_post[{sae_layer}]` "
            f"→ activations from `resid_pre[{int(layer)}]`"
        )
    else:
        sae_option = st.radio(
            "SAE",
            ["Hub — hook_z (Neuronpedia)", "Local — ln1 (trained)"],
            index=0,
        )
        if sae_option.startswith("Hub"):
            sae_type = "hub"
            hook_point = "attn.hook_z"
            sae_hub_release = "gpt2-small-hook-z-kk"
            sae_hub_id = f"blocks.{layer}.hook_z"
            supports_neuronpedia = True
            sae_local_path = ""
        else:
            sae_type = "local"
            hook_point = "ln1.hook_normalized"
            sae_hub_release = ""
            sae_hub_id = ""
            supports_neuronpedia = False
            default_local = str(
                Path(__file__).parent.parent / "checkpoints" / "q9sczrvl" / "50003968"
            )
            sae_local_path = st.text_input("Checkpoint path", value=default_local)
            if not Path(sae_local_path).exists():
                st.warning("Checkpoint not found. Train with `python train_sae.py`.")

    st.subheader("Compute settings")
    top_k_feat = st.slider("Top-K features / position", 5, 50, 20)
    # Large models / large d_sae — default to small chunks to avoid OOM
    default_chunk = 1 if (is_gemma or is_qwen) else 16
    chunk_size = st.slider("Chunk size (↓ = less GPU mem)", 1, 32, default_chunk)
    top_k_pairs = st.slider("Top-K pairs to display", 10, 100, 30)
    filter_self = st.checkbox("Filter self-interactions (q==k)", value=False)
    hide_bos = st.checkbox(
        "Hide BOS token in display",
        value=False,
        help=(
            "Some models (e.g. Gemma, Qwen) prepend a BOS token that dominates attention. "
            "Check this to exclude position 0 from all visualisations. "
            "BOS is always included in computation for correctness."
        ),
    )

    st.subheader("Ranking mode")
    agg_mode = st.radio(
        "Rank feature pairs by:",
        options=["avg", "sum", "max"],
        format_func={
            "avg": "(ii) Non-zero avg — mean strength when pair fires",
            "sum": "(i)  Sum — total strength over all positions",
            "max": "(iii) Max — strongest single occurrence",
        }.get,
        index=0,
        help=(
            "**(i) Sum**: total |FRA| summed over all position-pairs. Biased toward "
            "pairs that fire often.\n\n"
            "**(ii) Non-zero avg** (recommended): mean |FRA| divided only by the "
            "number of position-pairs where the pair actually fires. Best for "
            "finding cleanest semantic interactions.\n\n"
            "**(iii) Max**: the single highest |FRA| value anywhere in the sentence. "
            "Good for finding the strongest individual occurrence."
        ),
    )

    if is_gemma:
        hf_token = st.text_input(
            "HuggingFace token (for Gemma)",
            type="password",
            help="Required if you haven't run `huggingface-cli login`. Get yours at huggingface.co/settings/tokens",
        )
    elif is_qwen:
        hf_token = st.text_input(
            "HuggingFace token (optional)",
            type="password",
            help="Usually not required for Qwen2.5. Provide if you hit auth errors.",
        )
    else:
        hf_token = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

    compute_btn = st.button("▶  Compute FRA", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧠 Feature-Resolved Attention Dashboard")
st.caption("Decomposing attention through SAE feature space.")

# ---------------------------------------------------------------------------
# Trigger computation
# ---------------------------------------------------------------------------

if compute_btn:
    with st.spinner("Loading model & SAE…"):
        load_model(model_name, device, hf_token)
        if sae_type == "hub":
            load_sae_hub(sae_hub_release, sae_hub_id, device)
        elif sae_type == "gemma":
            load_sae_gemma(sae_hub_release, sae_hub_id, device)
        elif sae_type == "qwen":
            load_sae_qwen(sae_hub_release, sae_hub_id, device)
        elif Path(sae_local_path).exists():
            load_sae_local(sae_local_path, int(layer), device)

    head_arg = selected_heads[0] if len(selected_heads) == 1 else selected_heads
    n_heads_label = f"head {selected_heads[0]}" if len(selected_heads) == 1 else f"{len(selected_heads)} heads"
    with st.spinner(f"Computing FRA for {n_heads_label}…"):
        fra_data = run_fra(
            text=text,
            layer=int(layer),
            head=head_arg,
            hook_point=hook_point,
            sae_type=sae_type,
            sae_hub_release=sae_hub_release,
            sae_hub_id=sae_hub_id,
            sae_local_path=sae_local_path,
            top_k_features=top_k_feat,
            device=device,
            model_name=model_name,
            chunk_size=int(chunk_size),
            hf_token=hf_token,
        )

    st.session_state["fra_data"] = fra_data
    st.session_state["fra_config"] = {
        "layer": int(layer),
        "head": selected_heads[0],
        "text": text,
        "supports_neuronpedia": supports_neuronpedia,
        "np_model": np_model if is_qwen else "gpt2-small",
        "np_sae_suffix": np_sae_suffix if is_qwen else "att-kk",
        "np_layer": np_layer if is_qwen else int(layer),
        "filter_self": filter_self,
        "top_k_pairs": top_k_pairs,
        "agg_mode": agg_mode,
        "hide_bos": hide_bos,
        "trained_on_bos": not (is_gemma or is_qwen),
    }
    st.success(
        f"Done — {fra_data['total_interactions']:,} non-zero interactions found."
    )

# ---------------------------------------------------------------------------
# Main results area
# ---------------------------------------------------------------------------

if "fra_data" not in st.session_state:
    st.info("Configure the sidebar and click **▶ Compute FRA** to begin.")
    with st.expander("What is Feature-Resolved Attention?"):
        st.markdown(
            """
**Feature-Resolved Attention (FRA)** replaces the standard `[seq, seq]` attention
matrix with a `[seq, seq, d_sae, d_sae]` tensor, where each entry captures how much
**SAE query-feature _i_** at position _q_ attends to **SAE key-feature _j_** at
position _k_.

This lets us ask:
- Which _semantic_ features in the query attend strongly to which key features?
- Are there **conceptual induction heads** that copy specific concepts across positions?
- How do feature-level interactions differ from token-level ones?
"""
        )
    st.stop()

fra_data = st.session_state["fra_data"]
cfg = st.session_state["fra_config"]
layer_ = cfg["layer"]
head_ = cfg["head"]

# ── Multi-head switching (no recompute) ──────────────────────────────
if "per_head" in fra_data:
    available_heads = fra_data["heads"]
    selected_head = st.selectbox(
        "Browse head",
        available_heads,
        index=available_heads.index(head_) if head_ in available_heads else 0,
        help="Switch between pre-computed heads without recomputing.",
    )
    head_ = selected_head
    cfg["head"] = head_
    # Swap in the selected head's data for display
    hd = fra_data["per_head"][head_]
    fra_data["indices_np"] = hd["indices_np"]
    fra_data["values_np"] = hd["values_np"]
    fra_data["shape"] = hd["shape"]
    fra_data["total_interactions"] = hd["total_interactions"]
    fra_data["bias_corr_np"] = hd["bias_corr_np"]
    fra_data["attn_pattern_np"] = hd["attn_pattern_np"]

seq_len_full = fra_data["seq_len"]

# --- BOS filtering ---
# If the model prepended a BOS token and the user wants to hide it,
# we mask out position 0 from the sparse indices for display purposes.
_hide_bos = cfg.get("hide_bos", False) and fra_data.get("has_bos", False)
if _hide_bos:
    # Filter sparse entries: drop any row where q_pos==0 or k_pos==0,
    # then shift remaining positions down by 1.
    _idx = fra_data["indices_np"]
    _vals = fra_data["values_np"]
    _keep = (_idx[0] > 0) & (_idx[1] > 0)
    display_indices = _idx[:, _keep].copy()
    display_indices[0] -= 1  # shift q_pos
    display_indices[1] -= 1  # shift k_pos
    display_values = _vals[_keep]
    seq_len = seq_len_full - 1
    token_strs = fra_data["token_strs"][1:seq_len_full]
    feat_acts_display = fra_data["feat_acts_np"][1:seq_len_full]
    attn_pattern_display = fra_data["attn_pattern_np"][1:seq_len_full, 1:seq_len_full]
else:
    display_indices = fra_data["indices_np"]
    display_values = fra_data["values_np"]
    seq_len = seq_len_full
    token_strs = fra_data["token_strs"][:seq_len_full]
    feat_acts_display = fra_data["feat_acts_np"][:seq_len_full]
    attn_pattern_display = fra_data["attn_pattern_np"][:seq_len_full, :seq_len_full]

# Recompute pairs (filter / top_k / agg_mode may change without recomputing FRA)
agg_mode = cfg.get("agg_mode", "avg")
pairs = get_ranked_pairs(
    display_indices,
    display_values,
    top_k=cfg["top_k_pairs"],
    filter_self=cfg["filter_self"],
    mode=agg_mode,
)
bottom_pairs = get_bottom_pairs(
    display_indices,
    display_values,
    top_k=cfg["top_k_pairs"],
    filter_self=cfg["filter_self"],
)

# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens", seq_len)
c2.metric("Non-zero interactions", f"{fra_data['total_interactions']:,}")
total_unique = len(_aggregate_pairs(display_indices, display_values, cfg["filter_self"]))
c3.metric("Unique feature pairs", f"{total_unique:,}")
c4.metric(f"Layer / Head", f"L{layer_} / H{head_}")

if _hide_bos:
    st.caption("BOS token hidden from display (still included in computation).")

# Tokenised text display
tok_html = " ".join(
    f'<span style="background:#e9ecef;padding:2px 5px;border-radius:3px;'
    f'font-family:monospace;font-size:0.9em;">{html_lib.escape(t)}</span>'
    for t in token_strs
)
st.markdown(tok_html, unsafe_allow_html=True)
st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Top Interactions",
    "🔥 Feature Matrix",
    "🔍 Attention Comparison",
    "🔬 Ablation",
    "🔎 Feature Dashboard",
])

# ── Tab 1: Top / Least Interactions ────────────────────────────────────────

with tab1:
    agg_labels = {"avg": "(ii) Non-zero avg", "sum": "(i) Sum", "max": "(iii) Max"}
    st.caption(f"Ranking by: **{agg_labels.get(agg_mode, agg_mode)}** — change in sidebar.")

    rank_mode = st.radio(
        "Show:",
        ["Top interactions (strongest)", "Least interactions (weakest)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    active_pairs = pairs if rank_mode.startswith("Top") else bottom_pairs

    if not active_pairs:
        st.warning("No interactions found with current filters.")
    else:
        col_list, col_detail = st.columns([1, 2])

        with col_list:
            st.subheader("Feature pairs")
            # Label shows the active ranking metric
            def _pair_label(i):
                q, k, s, cnt, mx = active_pairs[i]
                avg_s = s / max(cnt, 1)
                if agg_mode == "avg":
                    score_str = f"avg={avg_s:.3f}"
                elif agg_mode == "max":
                    score_str = f"max={mx:.3f}"
                else:
                    score_str = f"sum={s:.3f}"
                suffix = "  ⟲" if q == k else ""
                return f"F{q}→F{k}  {score_str}{suffix}"

            selected_idx = st.radio(
                "Select a pair to inspect:",
                range(len(active_pairs)),
                format_func=_pair_label,
                label_visibility="collapsed",
            )

        with col_detail:
            q_sel, k_sel, strength_sel, count_sel, max_sel = active_pairs[selected_idx]
            avg_sel = strength_sel / max(count_sel, 1)
            is_self = q_sel == k_sel

            st.subheader(
                f"Feature {q_sel} → Feature {k_sel}"
                + ("  ⟲ self" if is_self else "")
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("(i) Sum |FRA|",   f"{strength_sel:.4f}")
            m2.metric("(ii) Non-zero avg", f"{avg_sel:.4f}", help=f"over {count_sel} position-pairs")
            m3.metric("(iii) Max |FRA|",  f"{max_sel:.4f}")
            if is_self:
                st.info(
                    "Self-interaction: query and key are the **same** feature. "
                    "This is a candidate for a **conceptual induction head** channel."
                )

            # --- Per-token activation bars ---
            feat_acts = feat_acts_display  # [seq_len, d_sae]
            q_acts = feat_acts[:, q_sel]
            k_acts = feat_acts[:, k_sel]

            barA, barB = st.columns(2)
            with barA:
                st.markdown(f"**Query feature {q_sel}** — token activations")
                st.plotly_chart(
                    token_activation_bar(
                        token_strs, q_acts, "rgba(102,126,234,0.75)"
                    ),
                    use_container_width=True,
                )
            with barB:
                st.markdown(f"**Key feature {k_sel}** — token activations")
                st.plotly_chart(
                    token_activation_bar(
                        token_strs, k_acts, "rgba(118,75,162,0.75)"
                    ),
                    use_container_width=True,
                )

            # --- Position heatmap: [seq, seq] for this pair ---
            st.markdown("**Position heatmap** — where does this pair interact?")
            pos_mat = get_position_heatmap(
                display_indices,
                display_values,
                q_sel, k_sel, seq_len,
            )
            _tick_vals = list(range(len(token_strs)))
            _tick_text = [html_lib.escape(t) for t in token_strs]
            fig_pos = go.Figure(go.Heatmap(
                z=pos_mat,
                x=_tick_vals,
                y=_tick_vals,
                colorscale="Blues",
                hovertemplate=(
                    "Q-pos: %{y}<br>K-pos: %{x}<br>Strength: %{z:.4f}"
                    "<extra></extra>"
                ),
            ))
            fig_pos.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Key token",
                yaxis_title="Query token",
                yaxis_autorange="reversed",
                xaxis=dict(tickvals=_tick_vals, ticktext=_tick_text),
                yaxis=dict(tickvals=_tick_vals, ticktext=_tick_text),
            )
            st.plotly_chart(fig_pos, use_container_width=True)

            # --- Neuronpedia iframes ---
            if cfg["supports_neuronpedia"]:
                _np_m = cfg.get("np_model", "gpt2-small")
                _np_s = cfg.get("np_sae_suffix", "att-kk")
                _np_l = cfg.get("np_layer", layer_)
                np_col1, np_col2 = st.columns(2)
                with np_col1:
                    desc_q = fetch_neuronpedia(_np_l, q_sel, _np_m, _np_s)
                    st.markdown(
                        f"**Neuronpedia — F{q_sel}:** _{desc_q}_"
                    )
                    st.components.v1.iframe(
                        neuronpedia_embed_url(_np_l, q_sel, _np_m, _np_s),
                        height=380,
                    )
                with np_col2:
                    desc_k = fetch_neuronpedia(_np_l, k_sel, _np_m, _np_s)
                    st.markdown(
                        f"**Neuronpedia — F{k_sel}:** _{desc_k}_"
                    )
                    st.components.v1.iframe(
                        neuronpedia_embed_url(_np_l, k_sel, _np_m, _np_s),
                        height=380,
                    )
            else:
                st.info(
                    "Neuronpedia is only available with the hub (hook_z) SAE "
                    "or Qwen2.5 SAEs. Switch SAE type in the sidebar to enable it."
                )

# ── Tab 2: Feature Matrix ──────────────────────────────────────────────────

with tab2:
    st.subheader(f"FRA Feature Interaction Matrix — L{layer_} H{head_}")
    mode_desc = {"avg": "non-zero average", "sum": "sum", "max": "max"}.get(agg_mode, agg_mode)
    st.caption(
        f"Each cell shows the **{mode_desc}** absolute interaction strength "
        "over all position pairs. Only features appearing in the ranked list are shown."
    )

    if not pairs:
        st.warning("No pairs to display.")
    else:
        # Collect unique features from top pairs
        top_features = []
        seen = set()
        for q, k, *_ in pairs:
            for f in (q, k):
                if f not in seen:
                    seen.add(f)
                    top_features.append(f)
            if len(top_features) >= 30:
                break

        feat_to_idx = {f: i for i, f in enumerate(top_features)}
        n = len(top_features)
        # Build matrix using the currently active ranking score
        matrix = np.zeros((n, n))
        for q, k, s, cnt, mx in pairs:
            if q in feat_to_idx and k in feat_to_idx:
                if agg_mode == "avg":
                    score = s / max(cnt, 1)
                elif agg_mode == "max":
                    score = mx
                else:
                    score = s
                matrix[feat_to_idx[q], feat_to_idx[k]] += score

        labels = [f"F{f}" for f in top_features]

        fig_mat = go.Figure(go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertemplate=(
                "Query: %{y}<br>Key: %{x}<br>Strength: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_mat.update_layout(
            height=600,
            xaxis_title="Key Feature",
            yaxis_title="Query Feature",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_mat, use_container_width=True)

# ── Tab 3: Attention Comparison ────────────────────────────────────────────

with tab3:
    st.subheader(f"Standard vs FRA Attention — L{layer_} H{head_}")

    col_std, col_fra = st.columns(2)

    with col_std:
        st.markdown("**Standard token-level attention** (post-softmax)")
        attn = attn_pattern_display
        _attn_tvals = list(range(len(token_strs)))
        _attn_ttext = [html_lib.escape(t) for t in token_strs]
        fig_attn = go.Figure(go.Heatmap(
            z=attn,
            x=_attn_tvals,
            y=_attn_tvals,
            colorscale="RdBu",
            hovertemplate=(
                "Q: %{y}<br>K: %{x}<br>Weight: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_attn.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Key",
            yaxis_title="Query",
            yaxis_autorange="reversed",
            xaxis=dict(tickvals=_attn_tvals, ticktext=_attn_ttext),
            yaxis=dict(tickvals=_attn_tvals, ticktext=_attn_ttext),
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    with col_fra:
        st.markdown(
            "**FRA attention** — reconstructed pre-softmax scores "
            "(bilinear + bias correction + softcap)"
        )
        # Reconstruct full attention scores with bias correction and softcap
        _bc = fra_data.get("bias_corr_np")
        _as = fra_data.get("attn_scale", 1.0)
        _sc = fra_data.get("softcap", 0.0)
        if _bc is not None:
            fra_pos_mat = get_fra_reconstructed_scores(
                display_indices, display_values, seq_len,
                _bc[1:, 1:] if _hide_bos else _bc,
                _as, _sc,
            )
        else:
            # Fallback: raw sum (no bias correction available)
            from fra.fra_func import _apply_softcap as _sc_fn
            fra_pos_mat = np.zeros((seq_len, seq_len))
            np.add.at(fra_pos_mat, (display_indices[0], display_indices[1]), display_values)
            fra_pos_mat = _sc_fn(fra_pos_mat / _as, _sc)

        _fra_tvals = list(range(len(token_strs)))
        _fra_ttext = [html_lib.escape(t) for t in token_strs]
        fig_fra_attn = go.Figure(go.Heatmap(
            z=fra_pos_mat,
            x=_fra_tvals,
            y=_fra_tvals,
            colorscale="RdBu",
            hovertemplate=(
                "Q: %{y}<br>K: %{x}<br>FRA score: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_fra_attn.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Key",
            yaxis_title="Query",
            yaxis_autorange="reversed",
            xaxis=dict(tickvals=_fra_tvals, ticktext=_fra_ttext),
            yaxis=dict(tickvals=_fra_tvals, ticktext=_fra_ttext),
        )
        st.plotly_chart(fig_fra_attn, use_container_width=True)

    st.info(
        "The FRA attention heatmap shows reconstructed pre-softmax attention scores: "
        "the feature-pair bilinear sum + b_dec bias correction, scaled and "
        "softcapped to match the model's actual attention logits."
    )

# ── Ablation helpers ─────────────────────────────────────────────────────


def parse_pair_list_json(json_str):
    """Parse ``[[q, k], ...]`` JSON into a list of (q, k) tuples or an error string."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    if not isinstance(data, list):
        return "Expected a JSON array of [q_feat, k_feat] pairs."
    pairs = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return f"Item {i} must be a 2-element array, got {item!r}."
        q, k = item
        if not (isinstance(q, int) and isinstance(k, int) and q >= 0 and k >= 0):
            return f"Item {i}: feature IDs must be non-negative integers, got {item!r}."
        pairs.append((q, k))
    # deduplicate preserving order
    seen = set()
    deduped = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def parse_feature_groups_json(json_str, include_self=False):
    """Parse ``{"name": [ids...], ...}`` and generate all within-group directed pairs.

    Returns ``(pairs, groups_dict)`` on success or an error string.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    if not isinstance(data, dict):
        return "Expected a JSON object mapping group names to feature ID arrays."
    groups = {}
    for name, ids in data.items():
        if not isinstance(ids, list):
            return f"Group '{name}': value must be an array of integers."
        for i, fid in enumerate(ids):
            if not isinstance(fid, int) or fid < 0:
                return f"Group '{name}', index {i}: must be a non-negative integer, got {fid!r}."
        groups[name] = ids

    all_pairs = set()
    for name, ids in groups.items():
        for q in ids:
            for k in ids:
                if q == k and not include_self:
                    continue
                all_pairs.add((q, k))
    return list(all_pairs), groups


def validate_pairs_against_tensor(pairs, indices_np):
    """Return (found, missing) lists of pairs based on what exists in the FRA tensor."""
    tensor_pairs = set(zip(indices_np[2].tolist(), indices_np[3].tolist()))
    found = [p for p in pairs if p in tensor_pairs]
    missing = [p for p in pairs if p not in tensor_pairs]
    return found, missing


def load_feature_embeddings(file_content):
    """Load pre-computed feature embeddings from a JSON or .npz file.

    Expected JSON format::

        {
            "embeddings": {<feat_id_str>: [float, ...], ...},
            "descriptions": {<feat_id_str>: "text", ...}   // optional
        }

    Or a numpy .npz with key ``embeddings`` of shape [n_features, embed_dim]
    and optional key ``feature_ids`` of shape [n_features].

    Returns (embeddings_dict, descriptions_dict) or an error string.
    ``embeddings_dict``: ``{int_feat_id: np.array}``
    ``descriptions_dict``: ``{int_feat_id: str}`` or empty dict.
    """
    # Try JSON first
    try:
        data = json.loads(file_content)
        if not isinstance(data, dict) or "embeddings" not in data:
            return "JSON must have an 'embeddings' key mapping feature IDs to vectors."
        emb_raw = data["embeddings"]
        embeddings = {int(k): np.array(v, dtype=np.float32) for k, v in emb_raw.items()}
        descriptions = {}
        if "descriptions" in data:
            descriptions = {int(k): str(v) for k, v in data["descriptions"].items()}
        return embeddings, descriptions
    except (json.JSONDecodeError, ValueError):
        return "Could not parse file as JSON. Expected format: {\"embeddings\": {\"feat_id\": [vec], ...}}"


def cluster_embeddings(embeddings_dict, feature_ids, n_clusters, method="agglomerative"):
    """Cluster a subset of features using their pre-computed embeddings.

    Args:
        embeddings_dict: {feat_id: np.array} from load_feature_embeddings.
        feature_ids: list of feature IDs to cluster (must be keys in embeddings_dict).
        n_clusters: number of clusters.
        method: "agglomerative" or "kmeans".

    Returns dict mapping cluster_label (int) -> list of feature IDs.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import normalize

    ids = [f for f in feature_ids if f in embeddings_dict]
    if len(ids) < 2:
        return {"cluster_0": ids}

    mat = np.stack([embeddings_dict[f] for f in ids])
    mat = normalize(mat)  # L2-normalize for cosine-like clustering

    n_clusters = min(n_clusters, len(ids))

    if method == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average",
        ).fit_predict(mat)
    else:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(mat)

    groups = defaultdict(list)
    for feat_id, label in zip(ids, labels):
        groups[f"cluster_{label}"].append(feat_id)
    return dict(groups)


# ── Tab 4: Ablation ──────────────────────────────────────────────────────

with tab4:
    st.subheader(f"Feature-Pair Ablation — L{layer_} H{head_}")
    st.caption(
        "Ablate selected feature pairs from the FRA tensor and measure the "
        "impact on model output. This reveals which cross-feature interactions "
        "are causally important to this attention head's computation."
    )

    st.info(
        "Ablation removes entire feature-pair **channels** from the 4D FRA tensor. "
        "For a selected pair (i, j), all entries FRA[:, :, i, j] across every position "
        "pair are zeroed out — the pair's interaction is removed everywhere in the sequence.",
        icon="ℹ️",
    )

    abl_exec = st.radio(
        "Ablation mode",
        ["Set (all at once)", "Individual (one pair at a time)", "Cumulative (accumulate pairs)"],
        horizontal=True,
        help=(
            "**Set**: ablate all selected pairs at once (fastest). "
            "**Individual**: ablate each pair alone, rank by causal impact (N forward passes). "
            "**Cumulative**: walk through pairs in ranked order, accumulating ablations (N forward passes)."
        ),
    )
    _exec_map = {
        "Set (all at once)": "set",
        "Individual (one pair at a time)": "individual",
        "Cumulative (accumulate pairs)": "cumulative",
    }
    exec_strategy = _exec_map[abl_exec]

    abl_strategy = st.radio(
        "Pair selection",
        ["Top-ranked pairs", "Feature-centric", "Custom pair set", "Feature-group ablation"],
        horizontal=True,
    )

    selected_pairs = []
    pairs_valid = False

    # ── Strategy 1: Top-ranked pairs (existing logic) ────────────────
    if abl_strategy == "Top-ranked pairs":
        all_pairs_for_ablation = get_ranked_pairs(
            display_indices, display_values,
            top_k=100, filter_self=False, mode=agg_mode,
        )
        if not all_pairs_for_ablation:
            st.warning("No feature pairs found. Compute FRA first.")
        else:
            offdiag_list = [p for p in all_pairs_for_ablation if p[0] != p[1]]
            ondiag_list = [p for p in all_pairs_for_ablation if p[0] == p[1]]
            st.markdown(f"**{len(offdiag_list)}** off-diagonal, "
                        f"**{len(ondiag_list)}** on-diagonal in top 100.")

            abl_col1, abl_col2 = st.columns(2)
            with abl_col1:
                n_ablate = st.slider(
                    "Number of top pairs to ablate",
                    min_value=1, max_value=min(50, len(offdiag_list) or 1),
                    value=min(10, len(offdiag_list) or 1),
                )
                abl_target = st.radio(
                    "Ablation target",
                    ["Top off-diagonal (i!=j)", "Top on-diagonal (i==j)", "Random off-diagonal"],
                    help=(
                        "**Off-diagonal**: cross-feature interactions. "
                        "**On-diagonal**: self-interactions. Random is a control."
                    ),
                )
            with abl_col2:
                if abl_target.startswith("Top off"):
                    selected_pairs = offdiag_list[:n_ablate]
                elif abl_target.startswith("Top on"):
                    selected_pairs = ondiag_list[:n_ablate]
                else:
                    import random as _random
                    _rng = _random.Random(42)
                    selected_pairs = _rng.sample(offdiag_list, min(n_ablate, len(offdiag_list)))
                pairs_valid = len(selected_pairs) > 0

    # ── Strategy 2: Feature-centric ─────────────────────────────────
    elif abl_strategy == "Feature-centric":
        abl_col1, abl_col2 = st.columns(2)
        with abl_col1:
            _target_feat = st.number_input(
                "Target feature ID", min_value=0, value=0,
                help="All pairs involving this feature will be selected.",
            )
            _feat_role = st.radio(
                "Feature role",
                ["Both (query or key)", "Query only", "Key only"],
                help="Whether the target feature appears as q_feat, k_feat, or either.",
            )
            _role_map = {
                "Both (query or key)": "both",
                "Query only": "query",
                "Key only": "key",
            }
            _role = _role_map[_feat_role]
            _max_feat_pairs = st.slider(
                "Max pairs", 5, 200, 50,
                help="Maximum number of pairs to select from ranking.",
            )
        with abl_col2:
            all_feat_pairs = get_ranked_pairs(
                display_indices, display_values,
                top_k=9999, filter_self=False, mode=agg_mode,
            )
            if _role == "query":
                feat_pairs = [p for p in all_feat_pairs if p[0] == _target_feat]
            elif _role == "key":
                feat_pairs = [p for p in all_feat_pairs if p[1] == _target_feat]
            else:
                feat_pairs = [p for p in all_feat_pairs
                              if p[0] == _target_feat or p[1] == _target_feat]

            if not feat_pairs:
                st.warning(f"No pairs found involving feature {_target_feat}.")
            else:
                st.success(
                    f"**{len(feat_pairs)}** pairs found for feature {_target_feat} "
                    f"(showing top {min(_max_feat_pairs, len(feat_pairs))})."
                )
                selected_pairs = feat_pairs[:_max_feat_pairs]
                pairs_valid = True

    # ── Strategy 3: Custom pair set ──────────────────────────────────
    elif abl_strategy == "Custom pair set":
        abl_col1, abl_col2 = st.columns(2)
        with abl_col1:
            st.caption("JSON format: `[[q_feat, k_feat], ...]`")
            _pair_json = st.text_area(
                "Pairs JSON",
                placeholder='[[5, 10], [3, 7], [12, 12]]',
                height=150,
                key="abl_pair_json",
            )
            _pair_file = st.file_uploader(
                "Or upload .json", type=["json"], key="abl_pairs_upload",
            )
        with abl_col2:
            raw_json = None
            if _pair_file is not None:
                raw_json = _pair_file.read().decode("utf-8")
                st.caption("Using uploaded file.")
            elif _pair_json.strip():
                raw_json = _pair_json

            if raw_json:
                result = parse_pair_list_json(raw_json)
                if isinstance(result, str):
                    st.error(result)
                else:
                    found, missing = validate_pairs_against_tensor(
                        result, fra_data["indices_np"],
                    )
                    if missing:
                        st.warning(
                            f"{len(missing)} of {len(result)} pairs not found in "
                            f"the FRA tensor (will have no effect)."
                        )
                    selected_pairs = [(q, k, 0.0, 0, 0.0) for q, k in result]
                    pairs_valid = len(selected_pairs) > 0
                    st.success(f"{len(result)} pairs parsed, {len(found)} present in tensor.")
            else:
                st.caption("Enter pairs or upload a file to continue.")

    # ── Strategy 4: Feature-group ablation ───────────────────────────
    elif abl_strategy == "Feature-group ablation":
        _group_source = st.radio(
            "Group source",
            ["Manual JSON", "Auto-cluster from embeddings file"],
            horizontal=True,
            help=(
                "**Manual**: provide groups as JSON. "
                "**Auto-cluster**: upload a pre-computed feature embeddings file "
                "(from auto-interp or Neuronpedia), and features active in this "
                "sample will be clustered automatically."
            ),
        )
        _include_self = st.checkbox("Include self-pairs (i, i)", value=False)

        if _group_source == "Manual JSON":
            abl_col1, abl_col2 = st.columns(2)
            with abl_col1:
                st.caption('JSON format: `{"group_name": [feat_ids...], ...}`')
                _group_json = st.text_area(
                    "Groups JSON",
                    placeholder='{"animals": [5, 10, 23], "verbs": [7, 42]}',
                    height=150,
                    key="abl_group_json",
                )
                _group_file = st.file_uploader(
                    "Or upload .json", type=["json"], key="abl_groups_upload",
                )
            with abl_col2:
                raw_json = None
                if _group_file is not None:
                    raw_json = _group_file.read().decode("utf-8")
                    st.caption("Using uploaded file.")
                elif _group_json.strip():
                    raw_json = _group_json

                if raw_json:
                    result = parse_feature_groups_json(raw_json, include_self=_include_self)
                    if isinstance(result, str):
                        st.error(result)
                    else:
                        pairs, groups = result
                        found, missing = validate_pairs_against_tensor(
                            pairs, fra_data["indices_np"],
                        )
                        if missing:
                            st.warning(
                                f"{len(missing)} of {len(pairs)} generated pairs not found "
                                f"in the FRA tensor."
                            )
                        for gname, gids in groups.items():
                            n = len(gids)
                            np_ = n * n if _include_self else n * (n - 1)
                            st.caption(f"**{gname}**: {n} features -> {np_} pairs")
                        if len(pairs) > 500:
                            st.warning(f"{len(pairs)} total pairs -- ablation may be slow.")
                        selected_pairs = [(q, k, 0.0, 0, 0.0) for q, k in pairs]
                        pairs_valid = len(selected_pairs) > 0
                        st.success(f"{len(pairs)} pairs generated, {len(found)} present in tensor.")
                else:
                    st.caption("Enter groups or upload a file to continue.")

        else:  # Auto-cluster from embeddings file
            abl_col1, abl_col2 = st.columns(2)
            with abl_col1:
                st.caption(
                    "Upload a JSON file with pre-computed feature embeddings "
                    "(e.g. from auto-interp or Neuronpedia description embeddings)."
                )
                st.code(
                    '{\n'
                    '  "embeddings": {"0": [0.1, ...], "5": [0.3, ...], ...},\n'
                    '  "descriptions": {"0": "articles", "5": "animals", ...}\n'
                    '}',
                    language="json",
                )
                _emb_file = st.file_uploader(
                    "Upload embeddings .json",
                    type=["json"],
                    key="abl_emb_upload",
                )
                _n_clusters = st.slider("Number of clusters", 2, 30, 8)
                _cluster_method = st.radio(
                    "Clustering method",
                    ["agglomerative", "kmeans"],
                    horizontal=True,
                )
                _ablate_clusters = st.multiselect(
                    "Clusters to ablate",
                    options=[],
                    help="Computed after uploading embeddings.",
                    key="abl_cluster_select",
                )

            with abl_col2:
                if _emb_file is not None:
                    emb_content = _emb_file.read().decode("utf-8")
                    emb_result = load_feature_embeddings(emb_content)
                    if isinstance(emb_result, str):
                        st.error(emb_result)
                    else:
                        emb_dict, desc_dict = emb_result
                        st.success(f"Loaded embeddings for {len(emb_dict)} features.")

                        # Get active features from the FRA tensor
                        active_feats = sorted(set(
                            fra_data["indices_np"][2].tolist()
                            + fra_data["indices_np"][3].tolist()
                        ))
                        covered = [f for f in active_feats if f in emb_dict]
                        not_covered = [f for f in active_feats if f not in emb_dict]

                        if not_covered:
                            st.warning(
                                f"{len(not_covered)} of {len(active_feats)} active features "
                                f"have no embedding — they will be excluded from clustering."
                            )

                        if len(covered) >= 2:
                            groups = cluster_embeddings(
                                emb_dict, covered, _n_clusters, _cluster_method,
                            )

                            # Display clusters with descriptions
                            cluster_names = sorted(groups.keys())
                            for cname in cluster_names:
                                feats = groups[cname]
                                # Show descriptions if available
                                descs = [desc_dict.get(f) for f in feats[:5] if f in desc_dict]
                                desc_str = ", ".join(d for d in descs if d)
                                label = f"**{cname}** ({len(feats)} features)"
                                if desc_str:
                                    label += f": _{desc_str}_"
                                st.caption(label)

                            # Let user pick which clusters to ablate
                            _ablate_clusters = st.multiselect(
                                "Clusters to ablate",
                                options=cluster_names,
                                default=[],
                                key="abl_cluster_select_live",
                            )

                            if _ablate_clusters:
                                # Merge selected clusters into groups dict for pair generation
                                merged = {}
                                for cname in _ablate_clusters:
                                    merged[cname] = groups[cname]
                                all_pairs = set()
                                for cname, feats in merged.items():
                                    for q in feats:
                                        for k in feats:
                                            if q == k and not _include_self:
                                                continue
                                            all_pairs.add((q, k))
                                pairs = list(all_pairs)
                                found, missing = validate_pairs_against_tensor(
                                    pairs, fra_data["indices_np"],
                                )
                                if missing:
                                    st.warning(
                                        f"{len(missing)} of {len(pairs)} pairs not in tensor."
                                    )
                                if len(pairs) > 500:
                                    st.warning(
                                        f"{len(pairs)} total pairs -- ablation may be slow."
                                    )
                                selected_pairs = [(q, k, 0.0, 0, 0.0) for q, k in pairs]
                                pairs_valid = len(selected_pairs) > 0
                                st.success(
                                    f"{len(pairs)} pairs from {len(_ablate_clusters)} cluster(s), "
                                    f"{len(found)} present in tensor."
                                )
                        else:
                            st.warning(
                                "Fewer than 2 active features have embeddings — "
                                "cannot cluster."
                            )
                else:
                    st.caption("Upload an embeddings file to continue.")

    # ── Unified pair display + run button ────────────────────────────
    if pairs_valid and selected_pairs:
        with st.expander(f"Selected pairs ({len(selected_pairs)})", expanded=False):
            for i, p in enumerate(selected_pairs[:20]):
                q, k = int(p[0]), int(p[1])
                marker = "self" if q == k else "cross"
                label = f"  F{q} -> F{k}  ({marker})"
                if len(p) >= 4 and p[2] > 0:
                    avg = p[2] / max(p[3], 1)
                    label += f"  avg={avg:.4f}, sum={p[2]:.4f}"
                st.text(label)
            if len(selected_pairs) > 20:
                st.text(f"  ... and {len(selected_pairs) - 20} more")

        if exec_strategy != "set":
            st.warning(
                f"**{exec_strategy.title()} mode** requires {len(selected_pairs)} "
                f"forward passes. This may take a while for large pair counts.",
                icon="⏱️",
            )

        run_abl = st.button("Run Ablation", type="primary")

        if run_abl:
            with st.spinner(f"Running {exec_strategy} ablation..."):
                from fra.ablation_study import (
                    ablate_fra_pairs,
                    compute_bias_corrections,
                    run_condition,
                )
                from fra.validation import fra_sum_to_attn

                # Get model and SAE from cache
                mdl = load_model(model_name, device, hf_token)
                if sae_type == "hub":
                    sae_obj = load_sae_hub(sae_hub_release, sae_hub_id, device)
                elif sae_type == "gemma":
                    sae_obj = load_sae_gemma(sae_hub_release, sae_hub_id, device)
                elif sae_type == "qwen":
                    sae_obj = load_sae_qwen(sae_hub_release, sae_hub_id, device)
                else:
                    sae_obj = load_sae_local(sae_local_path, int(layer_), device)

                # Bias corrections
                bias = compute_bias_corrections(
                    mdl, sae_obj, cfg["text"], layer_, head_, hook_point
                )

                if bias is None:
                    st.error("Text too short for ablation.")
                else:
                    # Ablation uses full (unfiltered) FRA tensor incl. BOS
                    abl_seq = seq_len_full
                    bias_seq = bias["seq_len"]
                    if bias_seq != abl_seq:
                        bias["term_q"] = bias["term_q"][:abl_seq]
                        bias["term_k"] = bias["term_k"][:abl_seq]
                        bias["seq_len"] = abl_seq
                        bias["tok_tensor"] = bias["tok_tensor"][:, :abl_seq]
                        bias["shift_labels"] = bias["tok_tensor"][0, 1:]
                        logits_trim = mdl(bias["tok_tensor"])
                        bias["unpatched_loss"] = F.cross_entropy(
                            logits_trim[0, :-1], bias["shift_labels"]
                        ).item()
                        bias["unpatched_logits"] = logits_trim

                    # Rebuild sparse tensor
                    d_sae_val = fra_data["feat_acts_np"].shape[1]
                    sp_indices = torch.tensor(fra_data["indices_np"], dtype=torch.long)
                    sp_values = torch.tensor(fra_data["values_np"], dtype=torch.float32)
                    sp_size = torch.Size([abl_seq, abl_seq, d_sae_val, d_sae_val])
                    fra_sparse = torch.sparse_coo_tensor(
                        sp_indices, sp_values, size=sp_size
                    ).coalesce()

                    # SAE scores baseline + FRA sum (shared across strategies)
                    sae_scores_np = bias["sae_scores"]
                    scores_full = torch.tensor(
                        sae_scores_np, dtype=torch.float32, device=device
                    )
                    fra_sum_full = fra_sum_to_attn(fra_sparse, abl_seq)
                    pairs_to_abl = [(int(p[0]), int(p[1])) for p in selected_pairs]
                    tok_t = bias["tok_tensor"]
                    shift_lab = bias["shift_labels"]
                    unp_logits = bias["unpatched_logits"]

                    # Shared baselines
                    r_full = run_condition(
                        mdl, layer_, head_, tok_t, shift_lab, scores_full, unp_logits
                    )
                    mask_t = torch.triu(
                        torch.full((abl_seq, abl_seq), float("-inf"), device=device),
                        diagonal=1,
                    )
                    scores_zero = torch.zeros((abl_seq, abl_seq), device=device) + mask_t
                    r_zero = run_condition(
                        mdl, layer_, head_, tok_t, shift_lab, scores_zero, unp_logits
                    )
                    hc = r_zero["loss"] - bias["unpatched_loss"]

                    # Helper: ablate a set of pairs using SAE baseline approach
                    def _ablate_set(pairs):
                        fra_abl = ablate_fra_pairs(fra_sparse, pairs, d_sae_val)
                        fra_sum_abl = fra_sum_to_attn(fra_abl, abl_seq)
                        delta = (fra_sum_full - fra_sum_abl) / bias["attn_scale"]
                        sc = sae_scores_np.copy()
                        sc -= delta
                        sc_t = torch.tensor(sc, dtype=torch.float32, device=device)
                        return run_condition(
                            mdl, layer_, head_, tok_t, shift_lab, sc_t, unp_logits
                        ), delta

                    # ────── SET ABLATION ──────────────────────────────
                    if exec_strategy == "set":
                        nnz_before = fra_sparse._nnz()
                        r_abl, delta = _ablate_set(pairs_to_abl)
                        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae_val)
                        n_removed = nnz_before - fra_ablated._nnz()

                        _lo_mask = np.tril(np.ones((abl_seq, abl_seq), dtype=bool))
                        delta_abs = np.abs(delta[_lo_mask])
                        sae_abs = np.abs(sae_scores_np[_lo_mask])
                        st.info(
                            f"**Set ablation**: pairs={len(pairs_to_abl)}, "
                            f"nnz removed={n_removed:,}/{nnz_before:,} "
                            f"({100*n_removed/max(nnz_before,1):.1f}%)  \n"
                            f"SAE scores range: [{sae_scores_np[_lo_mask].min():.2f}, "
                            f"{sae_scores_np[_lo_mask].max():.2f}]  \n"
                            f"Delta: mean={delta_abs.mean():.4f}, max={delta_abs.max():.4f}, "
                            f"**relative={100*delta_abs.sum()/max(sae_abs.sum(),1):.2f}%**",
                            icon="🔍",
                        )

                        st.markdown("---")
                        st.subheader("Set Ablation Results")

                        abl_vs_full = r_abl["loss"] - r_full["loss"]
                        st.caption(
                            f"Losses — unpatched: {bias['unpatched_loss']:.6f}, "
                            f"FRA full: {r_full['loss']:.6f}, ablated: {r_abl['loss']:.6f}, "
                            f"zero: {r_zero['loss']:.6f} | "
                            f"**ablated − full = {abl_vs_full:+.6f}**"
                        )
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Unpatched loss", f"{bias['unpatched_loss']:.4f}")
                        mc2.metric("FRA full loss", f"{r_full['loss']:.4f}",
                                   delta=f"{r_full['loss'] - bias['unpatched_loss']:+.4f}")
                        mc3.metric("Ablated loss", f"{r_abl['loss']:.4f}",
                                   delta=f"{r_abl['loss'] - bias['unpatched_loss']:+.4f}")
                        mc4.metric("Zero loss", f"{r_zero['loss']:.4f}",
                                   delta=f"{r_zero['loss'] - bias['unpatched_loss']:+.4f}")

                        mc5, mc6, mc7 = st.columns(3)
                        mc5.metric("KL div", f"{r_abl['kl_div']:.4f}")
                        mc6.metric("Top-1 changed",
                                   f"{r_abl['top1_change_frac']*100:.1f}%")
                        if hc > 0.01:
                            rec_full = (r_zero['loss'] - r_full['loss']) / hc
                            rec_abl = (r_zero['loss'] - r_abl['loss']) / hc
                            mc7.metric("Recovery (full→abl)",
                                       f"{rec_full:.3f} → {rec_abl:.3f}",
                                       delta=f"{rec_abl - rec_full:+.3f}")

                        # Heatmaps
                        st.markdown("**Attention score comparison**")
                        abl_token_strs = fra_data["token_strs"][:seq_len_full]
                        hm1, hm2, hm3 = st.columns(3)

                        def _score_heatmap(scores_np, title):
                            disp = scores_np.copy()
                            disp[np.triu_indices_from(disp, k=1)] = np.nan
                            _tv = list(range(len(abl_token_strs)))
                            _tt = [html_lib.escape(t) for t in abl_token_strs]
                            fig = go.Figure(go.Heatmap(
                                z=disp, x=_tv, y=_tv,
                                colorscale="RdBu", zmid=0,
                                hovertemplate=(
                                    "Q: %{y}<br>K: %{x}<br>"
                                    "Score: %{z:.2f}<extra></extra>"
                                ),
                            ))
                            fig.update_layout(
                                title=title, height=350,
                                margin=dict(l=0, r=0, t=30, b=0),
                                yaxis_autorange="reversed",
                                xaxis=dict(tickvals=_tv, ticktext=_tt),
                                yaxis=dict(tickvals=_tv, ticktext=_tt),
                            )
                            return fig

                        with hm1:
                            st.plotly_chart(
                                _score_heatmap(
                                    scores_full.cpu().numpy(), "FRA Full"
                                ),
                                use_container_width=True,
                            )
                        with hm2:
                            scores_abl_np = sae_scores_np.copy()
                            scores_abl_np -= delta
                            st.plotly_chart(
                                _score_heatmap(scores_abl_np, "After Ablation"),
                                use_container_width=True,
                            )
                        with hm3:
                            diff_np = scores_abl_np - scores_full.cpu().numpy()
                            st.plotly_chart(
                                _score_heatmap(diff_np, "Difference"),
                                use_container_width=True,
                            )

                    # ────── INDIVIDUAL ABLATION ───────────────────────
                    elif exec_strategy == "individual":
                        st.markdown("---")
                        st.subheader(
                            "Individual Ablation — pair-by-pair causal impact"
                        )
                        progress = st.progress(
                            0, text="Ablating pairs individually..."
                        )

                        indiv_results = []
                        for i, (q, k) in enumerate(pairs_to_abl):
                            progress.progress(
                                (i + 1) / len(pairs_to_abl),
                                text=f"Pair {i+1}/{len(pairs_to_abl)}: "
                                     f"F{q}→F{k}",
                            )
                            r, _ = _ablate_set([(q, k)])
                            indiv_results.append({
                                "q_feat": q, "k_feat": k,
                                "loss": r["loss"],
                                "loss_delta": r["loss"] - r_full["loss"],
                                "kl_div": r["kl_div"],
                                "top1_pct": r["top1_change_frac"] * 100,
                            })
                        progress.empty()

                        # Sort by |loss_delta|
                        indiv_results.sort(
                            key=lambda x: abs(x["loss_delta"]), reverse=True
                        )

                        st.caption(
                            f"Baselines — unpatched: {bias['unpatched_loss']:.4f}, "
                            f"FRA full: {r_full['loss']:.4f}, "
                            f"zero: {r_zero['loss']:.4f}, "
                            f"head contrib: {hc:+.4f}"
                        )

                        # Bar chart
                        top_n = min(30, len(indiv_results))
                        show = indiv_results[:top_n]
                        labels = [
                            f"F{r['q_feat']}→F{r['k_feat']}" for r in show
                        ]
                        deltas = [r["loss_delta"] for r in show]

                        fig_bar = go.Figure(go.Bar(
                            x=labels, y=deltas,
                            marker_color=[
                                "rgba(220,60,60,0.7)" if d > 0
                                else "rgba(60,120,220,0.7)"
                                for d in deltas
                            ],
                            hovertemplate=(
                                "%{x}<br>Δloss: %{y:+.6f}<extra></extra>"
                            ),
                        ))
                        fig_bar.update_layout(
                            height=350,
                            xaxis_title="Feature pair",
                            yaxis_title="Loss delta (ablated − full)",
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Table
                        import pandas as pd
                        df = pd.DataFrame(indiv_results)
                        df.index = range(1, len(df) + 1)
                        df.index.name = "Rank"
                        df.columns = [
                            "Q feat", "K feat", "Loss",
                            "Δ Loss", "KL div", "Top-1 %",
                        ]
                        st.dataframe(
                            df.style.format({
                                "Loss": "{:.6f}",
                                "Δ Loss": "{:+.6f}",
                                "KL div": "{:.6f}",
                                "Top-1 %": "{:.1f}",
                            }),
                            use_container_width=True,
                        )

                    # ────── CUMULATIVE ABLATION ───────────────────────
                    elif exec_strategy == "cumulative":
                        st.markdown("---")
                        st.subheader(
                            "Cumulative Ablation — accumulating pairs"
                        )
                        progress = st.progress(
                            0, text="Accumulating pairs..."
                        )

                        cum_results = []
                        ablated_so_far = []
                        for i, (q, k) in enumerate(pairs_to_abl):
                            progress.progress(
                                (i + 1) / len(pairs_to_abl),
                                text=f"Step {i+1}/{len(pairs_to_abl)}: "
                                     f"+F{q}→F{k}",
                            )
                            ablated_so_far.append((q, k))
                            r, _ = _ablate_set(list(ablated_so_far))
                            prev_loss = (
                                cum_results[-1]["loss"]
                                if cum_results
                                else r_full["loss"]
                            )
                            cum_results.append({
                                "step": i + 1,
                                "pair_added": f"F{q}→F{k}",
                                "n_ablated": len(ablated_so_far),
                                "loss": r["loss"],
                                "loss_delta": r["loss"] - r_full["loss"],
                                "marginal": r["loss"] - prev_loss,
                                "kl_div": r["kl_div"],
                                "top1_pct": r["top1_change_frac"] * 100,
                            })
                        progress.empty()

                        st.caption(
                            f"Baselines — unpatched: {bias['unpatched_loss']:.4f}, "
                            f"FRA full: {r_full['loss']:.4f}, "
                            f"zero: {r_zero['loss']:.4f}, "
                            f"head contrib: {hc:+.4f}"
                        )

                        # Loss curve
                        steps = [0] + [r["step"] for r in cum_results]
                        losses = [r_full["loss"]] + [
                            r["loss"] for r in cum_results
                        ]
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(
                            x=steps, y=losses, mode="lines+markers",
                            name="Loss",
                            hovertemplate=(
                                "Step %{x}: loss=%{y:.4f}<extra></extra>"
                            ),
                        ))
                        fig_line.add_hline(
                            y=bias["unpatched_loss"],
                            line_dash="dash", line_color="green",
                            annotation_text="Unpatched",
                        )
                        fig_line.add_hline(
                            y=r_zero["loss"],
                            line_dash="dash", line_color="red",
                            annotation_text="Zero-ablated",
                        )
                        fig_line.update_layout(
                            height=350,
                            xaxis_title="Pairs ablated",
                            yaxis_title="Loss",
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                        # Marginal bar chart
                        st.markdown(
                            "**Marginal loss change per added pair**"
                        )
                        fig_marg = go.Figure(go.Bar(
                            x=[r["pair_added"] for r in cum_results],
                            y=[r["marginal"] for r in cum_results],
                            marker_color=[
                                "rgba(220,60,60,0.7)" if r["marginal"] > 0
                                else "rgba(60,120,220,0.7)"
                                for r in cum_results
                            ],
                            hovertemplate=(
                                "%{x}<br>Marginal: %{y:+.6f}"
                                "<extra></extra>"
                            ),
                        ))
                        fig_marg.update_layout(
                            height=300,
                            xaxis_title="Pair added",
                            yaxis_title="Marginal Δ loss",
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig_marg, use_container_width=True)

                        # Table
                        import pandas as pd
                        df = pd.DataFrame(cum_results)
                        df.columns = [
                            "Step", "Pair added", "N ablated", "Loss",
                            "Δ Loss", "Marginal", "KL div", "Top-1 %",
                        ]
                        st.dataframe(
                            df.style.format({
                                "Loss": "{:.6f}",
                                "Δ Loss": "{:+.6f}",
                                "Marginal": "{:+.6f}",
                                "KL div": "{:.6f}",
                                "Top-1 %": "{:.1f}",
                            }),
                            use_container_width=True,
                        )

# ── Tab 5: Feature Dashboard (sae_vis) ──────────────────────────────────

with tab5:
    st.subheader("Feature Dashboard")
    st.caption(
        "Interactive feature visualization powered by "
        "[sae_vis](https://github.com/callummcdougall/sae_vis). "
        "Select top features from the FRA analysis to inspect their "
        "activation patterns, logits, and top examples."
    )

    _sae_vis_available = False
    _sae_vis_error = ""
    try:
        from sae_vis.data_fetching_fns import get_feature_data
        from sae_vis.data_config_classes import SaeVisConfig
        _sae_vis_available = True
    except ImportError as e:
        _sae_vis_error = str(e)
    except Exception as e:
        _sae_vis_error = str(e)

    if not _sae_vis_available:
        if "sae_lens" in _sae_vis_error:
            st.warning(
                "`sae_vis` is installed but cannot load because `sae_lens` "
                "is missing. Install both:\n\n"
                "```\npip install sae-vis sae-lens\n```\n\n"
                "Then restart the dashboard."
            )
        elif "sae_vis" in _sae_vis_error or not _sae_vis_error:
            st.warning(
                "`sae_vis` is not installed. Install it with:\n\n"
                "```\npip install sae-vis sae-lens\n```\n\n"
                "Then restart the dashboard."
            )
        else:
            _missing = _sae_vis_error.replace("No module named ", "").strip("'\"")
            st.warning(
                f"`sae_vis` failed to load — missing dependency `{_missing}`. "
                f"Run this in the **same** Python that runs the dashboard:\n\n"
                f"```\npip install {_missing}\n```\n\n"
                "Then restart the dashboard."
            )
    else:
        # Collect top features from FRA pairs
        _top_feats = set()
        for q, k, *_ in pairs[:20]:
            _top_feats.add(int(q))
            _top_feats.add(int(k))
        _top_feats_sorted = sorted(_top_feats)

        if not _top_feats_sorted:
            st.info("Run FRA first to populate top features.")
        else:
            _n_vis = st.slider(
                "Features to visualize", 1, min(20, len(_top_feats_sorted)),
                value=min(5, len(_top_feats_sorted)),
                help="More features = longer computation time.",
            )
            _selected_feats = _top_feats_sorted[:_n_vis]
            st.write(f"Features: {_selected_feats}")

            _run_vis = st.button("Generate Feature Dashboard", type="primary")

            if _run_vis:
                with st.spinner("Computing feature visualizations (this may take a minute)..."):
                    import tempfile

                    # Load model (reuses cached version)
                    _vis_model = load_model(model_name, device, hf_token)

                    # Load SAE
                    if sae_type == "hub":
                        _vis_sae_obj = load_sae_hub(sae_hub_release, sae_hub_id, device)
                    elif sae_type == "gemma":
                        _vis_sae_obj = load_sae_gemma(sae_hub_release, sae_hub_id, device)
                    else:
                        _vis_sae_obj = load_sae_local(sae_local_path, int(layer_), device)

                    # Get the underlying SAE Lens SAE object
                    _vis_sae = _vis_sae_obj.sae if hasattr(_vis_sae_obj, "sae") else _vis_sae_obj

                    # Patch hook_name onto config if missing — sae_vis expects
                    # the old SAE Lens config format with hook_name as a direct
                    # attribute, but newer sae_lens stores it in metadata.
                    if not hasattr(_vis_sae.cfg, "hook_name"):
                        _hook = f"blocks.{layer_}.{hook_point}"
                        # Try to get from metadata first
                        _meta = getattr(_vis_sae.cfg, "metadata", None)
                        if _meta and hasattr(_meta, "hook_name"):
                            _hook = _meta.hook_name
                        _vis_sae.cfg.hook_name = _hook
                    if not hasattr(_vis_sae.cfg, "hook_layer"):
                        _vis_sae.cfg.hook_layer = layer_
                    if not hasattr(_vis_sae.cfg, "hook_head_index"):
                        _vis_sae.cfg.hook_head_index = None

                    # Tokenize text for sae_vis
                    _vis_tokens = _vis_model.tokenizer.encode(cfg["text"])[:128]
                    _vis_tok_t = torch.tensor([_vis_tokens], device=device)

                    # Generate visualization data
                    _vis_cfg = SaeVisConfig(features=_selected_feats)
                    _vis_data = get_feature_data(
                        sae=_vis_sae,
                        model=_vis_model,
                        tokens=_vis_tok_t,
                        cfg=_vis_cfg,
                    )

                    # Save to temp file and read back HTML
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        _vis_data.save_feature_centric_vis(f.name, feature=_selected_feats[0])
                        _tmp_path = f.name

                    with open(_tmp_path, "r") as f:
                        _vis_html = f.read()

                    st.session_state["sae_vis_html"] = _vis_html
                    st.session_state["sae_vis_features"] = _selected_feats

            # Display cached HTML if available
            if "sae_vis_html" in st.session_state:
                st.markdown(
                    f"Showing dashboard for features: "
                    f"{st.session_state.get('sae_vis_features', [])}"
                )
                st.components.v1.html(
                    st.session_state["sae_vis_html"],
                    height=800,
                    scrolling=True,
                )