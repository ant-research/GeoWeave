"""RL operations adapter for GeoWeave (NEO-Unify MoT).

Thin adapter layer bridging the vendored NEOChatModel to UniRL's RL runtime.
Unlike Bagel's rl_ops, GeoWeave's ``_t2i_predict_v`` is NOT decorated with
``@torch.no_grad``, so we can call it directly with gradients flowing.

All functions take the NEOChatModel (``bundle.model``) as the first argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

@dataclass
class PolicyEntropyAccumulator:
    """Detached exact categorical entropy accumulated over scored policy tokens."""

    total: Optional[Tensor] = None
    count: int = 0

    def add_logits(self, logits: Tensor) -> None:
        """Accumulate ``H(softmax(logits))`` without extending the grad graph."""
        if logits.numel() == 0:
            return
        with torch.no_grad():
            scores = logits.detach().float()
            probs = torch.softmax(scores, dim=-1)
            entropy = torch.logsumexp(scores, dim=-1) - probs.mul_(scores).sum(dim=-1)
            chunk_sum = entropy.sum()
            self.total = chunk_sum if self.total is None else self.total + chunk_sum
            self.count += int(scores.shape[0])

    def mean(self) -> Optional[Tensor]:
        if self.total is None or self.count == 0:
            return None
        return self.total / float(self.count)


# ---------------------------------------------------------------------------
# Text prefix & query building
# ---------------------------------------------------------------------------


def build_query(
    model: Any,
    prompt: str,
    *,
    system_message: Optional[str] = None,
    append_text: Optional[str] = None,
) -> str:
    """Format a prompt with the chat template. Wraps ``model._build_t2i_query``."""
    return model._build_t2i_query(
        prompt, system_message=system_message, append_text=append_text
    )


def prepare_input_image(
    model: Any,
    image: Any,
) -> Tuple[Tensor, Tensor]:
    """Preprocess a PIL image into ``(pixel_values, grid_hw)`` tensors on device."""
    from .vendor.utils import load_image_native

    device = next(model.parameters()).device
    pv, ghw = load_image_native(
        image,
        patch_size=model.patch_size,
        downsample_ratio=model.downsample_ratio,
    )
    return pv.to(device=device, dtype=torch.bfloat16), ghw.to(device)


def insert_image_tokens(
    query: str,
    grid_hw_list: List[Tensor],
    downsample_ratio: float,
) -> str:
    """Replace each ``<image>`` placeholder in *query* with
    ``<img><IMG_CONTEXT>*N</img>`` where N is derived from *grid_hw_list*."""
    for ghw in grid_hw_list:
        n = int(ghw[0, 0].item() * ghw[0, 1].item() * downsample_ratio ** 2)
        tokens = "<img>" + "<IMG_CONTEXT>" * n + "</img>"
        query = query.replace("<image>", tokens, 1)
    return query


def build_text_inputs(
    model: Any,
    tokenizer: Any,
    query: str,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Tokenize a query and build 3D indexes + attention mask.

    Returns ``(input_ids [1, L], indexes [3, L], attention_mask_dict)``.
    Wraps ``model._build_t2i_text_inputs``.
    """
    return model._build_t2i_text_inputs(tokenizer, query)


def build_it2i_inputs(
    model: Any,
    tokenizer: Any,
    query: str,
    pixel_values: Optional[Tensor] = None,
    grid_hw: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Build inputs with optional image ViT embedding replacement.

    Returns ``(input_embeds [1, L, C], indexes [3, L], attention_mask_dict)``.
    Wraps ``model._build_it2i_inputs``.
    """
    return model._build_it2i_inputs(tokenizer, query, pixel_values, grid_hw)


# ---------------------------------------------------------------------------
# Text prefix forward
# ---------------------------------------------------------------------------


def prefix_forward(
    model: Any,
    input_ids: Tensor,
    indexes: Tensor,
    attention_mask: Any,
) -> Tuple[Any, Tensor]:
    """Text prefix forward → (past_key_values, last_hidden_state).

    Calls ``model._t2i_prefix_forward``. Grad follows ambient context.
    """
    return model._t2i_prefix_forward(input_ids, indexes, attention_mask)


def prefix_forward_embeds(
    model: Any,
    input_embeds: Tensor,
    indexes: Tensor,
    attention_mask: Any,
    gen_indicators: Optional[Tensor] = None,
) -> Tuple[Any, Tensor]:
    """Prefix forward from pre-built embeddings → (past_key_values, last_hidden_state).

    Calls ``model._it2i_prefix_forward``.
    """
    return model._it2i_prefix_forward(
        input_embeds, indexes, attention_mask, gen_indicators
    )


# ---------------------------------------------------------------------------
# Diffusion: image embedding & velocity prediction
# ---------------------------------------------------------------------------


def build_image_embeds(
    model: Any,
    noisy_image: Tensor,
    t: Tensor,
    *,
    grid_hw: Tensor,
    noise_scale_value: Optional[float] = None,
) -> Tensor:
    """Patchify noisy image → gen ViT → + timestep embed → image_embeds.

    Args:
        noisy_image: ``[B, 3, H, W]`` pixel-space noisy image.
        t: scalar timestep tensor.
        grid_hw: ``[B, 2]`` RAW patch grid dimensions for the gen ViT.
        noise_scale_value: optional noise scale for the noise_scale_embedder.

    Returns:
        ``image_embeds [B, tokens, hidden_dim]`` ready for LLM gen-branch forward.
    """
    patch_size = model.patch_size
    # Patchify with channel_first=True for ViT input
    patches = model.patchify(noisy_image, patch_size, channel_first=True)  # [B, L, patch_dim]
    # Gen ViT asserts pixel_values.dim()==2 (native-resolution path). Flatten
    # batch × L for the forward, then reshape the merged tokens back to [B, ...].
    # Mirrors vendor modeling_neo_chat.py:897.
    B = patches.shape[0]
    patches_2d = patches.reshape(-1, patches.shape[-1])
    image_embeds = model.extract_feature(patches_2d, gen_model=True, grid_hw=grid_hw)
    image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1])

    # Match the vendor batch path: embed one timestep per image token.
    token_count = image_embeds.shape[1]
    if t.numel() == 1:
        t_expanded = t.reshape(1).expand(B * token_count)
    elif t.numel() == B:
        t_expanded = t.reshape(B, 1).expand(B, token_count).reshape(-1)
    else:
        raise ValueError(f"t must be scalar or batch-sized, got shape={tuple(t.shape)}")
    timestep_embeds = model.fm_modules["timestep_embedder"](
        t_expanded
    ).reshape(B, token_count, -1)
    image_embeds = image_embeds + timestep_embeds

    # Optional noise-scale embedding (normalized by max_value per vendor)
    if noise_scale_value is not None and model.add_noise_scale_embedding:
        max_val = getattr(model, "noise_scale_max_value", 10.0)
        ns_tensor = torch.full_like(t_expanded, noise_scale_value / max_val)
        ns_embeds = model.fm_modules["noise_scale_embedder"](
            ns_tensor
        ).reshape(B, token_count, -1)
        image_embeds = image_embeds + ns_embeds

    return image_embeds


def build_image_indexes(
    model: Any,
    token_h: int,
    token_w: int,
    text_len: int,
    device: torch.device,
) -> Tensor:
    """Build 3D indexes (t, h, w) for image tokens.

    Wraps ``model._build_t2i_image_indexes``.
    Returns ``[3, token_h * token_w]``.
    """
    return model._build_t2i_image_indexes(token_h, token_w, text_len, device)


def predict_v(
    model: Any,
    input_embeds: Tensor,
    indexes_image: Tensor,
    attn_mask: Any,
    past_key_values: Any,
    t: Tensor,
    z: Tensor,
    image_token_num: int,
    *,
    timestep_embeddings: Optional[Tensor] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """Predict velocity via ``model._t2i_predict_v``. Already grad-capable.

    Returns ``v_pred [B, L, patch_dim]``.
    """
    return model._t2i_predict_v(
        input_embeds,
        indexes_image,
        attn_mask,
        past_key_values,
        t,
        z,
        image_token_num,
        timestep_embeddings=timestep_embeddings,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# Patchify / unpatchify / schedule
# ---------------------------------------------------------------------------


def patchify(model: Any, images: Tensor, patch_size: int, channel_first: bool = False) -> Tensor:
    return model.patchify(images, patch_size, channel_first=channel_first)


def unpatchify(model: Any, x: Tensor, patch_size: int, h: int, w: int) -> Tensor:
    return model.unpatchify(x, patch_size, h=h, w=w)


def apply_time_schedule(
    model: Any,
    timesteps: Tensor,
    image_seq_len: int,
    timestep_shift: float = 1.0,
) -> Tensor:
    """Apply time-schedule shifting. Wraps ``model._apply_time_schedule``."""
    return model._apply_time_schedule(timesteps, image_seq_len, timestep_shift)


def euler_step(v_pred: Tensor, z: Tensor, t: float, t_next: float) -> Tensor:
    """Simple Euler step: ``z_next = z + (t_next - t) * v_pred``."""
    return z + (t_next - t) * v_pred


# ---------------------------------------------------------------------------
# AR: per-token text decoding with log-prob emission
# ---------------------------------------------------------------------------


def decode_text(
    model: Any,
    tokenizer: Any,
    past_key_values: Any,
    t_idx: int,
    *,
    start_logits: Tensor,
    sample_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
    max_new_tokens: int,
    stop_ids: List[int],
    device: torch.device,
) -> Tuple[List[int], List[float], Any, int]:
    """Per-token AR decode with log-prob emission.

    Fixed iteration count for FSDP safety (same pattern as Bagel's decode_text).
    Recording stops at first stop token but forwards continue.

    Args:
        model: NEOChatModel.
        tokenizer: tokenizer (unused here but kept for API symmetry).
        past_key_values: KV cache from prefix forward.
        t_idx: current temporal index for RoPE.
        start_logits: logits from the prefix forward ``[1, vocab]``.
        sample_fn: ``(logits [1, V]) -> (token_id [1], log_prob [1])``
        max_new_tokens: fixed iteration count.
        stop_ids: list of stop token IDs (EOS, img_start, etc.).
        device: torch device.

    Returns:
        ``(tokens, log_probs, past_key_values, t_idx)``
        where tokens/log_probs include up to and including the stop token.
    """
    tokens: List[int] = []
    log_probs: List[float] = []
    recording = True

    # First token from prefix logits
    token_id, log_prob = sample_fn(start_logits[:, -1:, :].squeeze(0))
    next_token = token_id.view(1)

    for step in range(max_new_tokens):
        if recording:
            tid = next_token.item()
            tokens.append(tid)
            log_probs.append(log_prob.item())
            if tid in stop_ids:
                recording = False

        # Forward one token
        model.language_model.model.current_index = t_idx
        outputs = model.language_model(
            input_ids=next_token.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        t_idx += 1

        if recording or step < max_new_tokens - 1:
            token_id, log_prob = sample_fn(outputs.logits[:, -1, :])
            next_token = token_id.view(1)

    return tokens, log_probs, past_key_values, t_idx


# ---------------------------------------------------------------------------
# AR: teacher-forced replay scoring
# ---------------------------------------------------------------------------


def score_response(
    model: Any,
    *,
    input_ids: Tensor,
    response_ids: Tensor,
    indexes: Tensor,
    attention_mask: Any,
    temperature: float = 1.0,
    logprob_chunk: int = 4096,
    device: torch.device,
    entropy_accumulator: Optional[PolicyEntropyAccumulator] = None,
) -> Tensor:
    """One-shot teacher-forced replay: compute per-token log-probs for response.

    Forwards ``[query + response[:-1]]`` through the LLM und branch and gathers
    log-probs at response positions. Chunked lm_head with gradient checkpointing.

    Args:
        input_ids: ``[1, query_len + response_len - 1]`` full input sequence.
        response_ids: ``[response_len]`` the response tokens to score.
        indexes: ``[3, seq_len]`` 3D indexes for the full input.
        attention_mask: attention mask dict for the full input.
        temperature: sampling temperature for log-prob computation.
        logprob_chunk: chunk size for lm_head (memory optimization).
        device: torch device.

    Returns:
        ``[response_len]`` per-token log-probs in fp32.
    """
    n_response = response_ids.shape[0]

    # Forward through LLM (und branch: no image_gen_indicators)
    outputs = model.language_model.model(
        input_ids=input_ids,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # Extract hidden states at the response-predicting positions
    # (last n_response positions of the output predict response tokens)
    hidden = outputs.last_hidden_state[0, -n_response:]  # [n_response, C]

    # Chunked log-prob computation
    lm_head = model.language_model.lm_head
    all_logps = []
    for i in range(0, n_response, logprob_chunk):
        end = min(i + logprob_chunk, n_response)
        chunk_h = hidden[i:end]  # [chunk, C]
        if chunk_h.requires_grad:
            logits = torch.utils.checkpoint.checkpoint(
                lm_head, chunk_h, use_reentrant=False
            )
        else:
            logits = lm_head(chunk_h)
        logits = logits.float() / temperature  # [chunk, V]
        if entropy_accumulator is not None:
            entropy_accumulator.add_logits(logits)
        target = response_ids[i:end]  # [chunk]
        log_probs = logits.gather(1, target.unsqueeze(1)).squeeze(1) - torch.logsumexp(
            logits, dim=1
        )
        all_logps.append(log_probs)

    return torch.cat(all_logps, dim=0)  # [n_response]


# ---------------------------------------------------------------------------
# Packed interleave replay
# ---------------------------------------------------------------------------


@dataclass
class PackedInterleavePlan:
    """One teacher-forced understanding sequence for an interleaved response.

    ``prediction_positions[k]`` is the packed hidden-state position that predicts
    ``text_targets[k]``. ``image_boundary_*[j]`` describes the prefix immediately
    after text segment ``j`` and before generated image ``j`` is re-encoded.
    """

    inputs_embeds: Tensor
    indexes: Tensor
    attention_mask: Dict[str, Tensor]
    prediction_positions: Tensor
    text_targets: Tensor
    image_boundary_seq_lens: List[int]
    image_boundary_t_idxs: List[int]
    image_shapes: List[Tuple[int, int]]


@dataclass
class FunctionalKVLayer:
    keys: Tensor
    values: Tensor


@dataclass
class FunctionalKVCache:
    """Minimal read-only Cache protocol consumed by generation-branch attention."""

    layers: List[FunctionalKVLayer]

    def get_seq_length(self) -> int:
        if not self.layers:
            return 0
        return int(self.layers[0].keys.shape[2])


def build_reencoded_image_inputs(
    model: Any,
    tokenizer: Any,
    t_idx: int,
    image_tensor: Tensor,
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    """Build understanding-ViT embeds/indexes for one generated image.

    This is the functional (no KV mutation) half of :func:`append_image_to_cache`.
    It returns ``(inputs_embeds [1,N+1,C], indexes [3,N+1], new_t_idx)``.
    """
    from .vendor.modeling_neo_chat import build_abs_positions_from_grid_hw

    pred_img = image_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    raw_img = pred_img * 0.5 + 0.5
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    und_img = (raw_img - img_mean) / img_std

    ps = model.patch_size
    merge_size = int(1 / model.downsample_ratio)
    _, c, h_px, w_px = und_img.shape
    p_grid_h = h_px // ps
    p_grid_w = w_px // ps
    flatten_pv = und_img[0].view(c, p_grid_h, ps, p_grid_w, ps)
    flatten_pv = flatten_pv.permute(1, 3, 0, 2, 4).reshape(p_grid_h * p_grid_w, c * ps * ps)

    raw_grid_hw = torch.tensor([[p_grid_h, p_grid_w]], device=device, dtype=torch.long)
    merged_grid_hw = torch.tensor(
        [[p_grid_h // merge_size, p_grid_w // merge_size]], device=device, dtype=torch.long
    )
    vit_embeds = model.extract_feature(flatten_pv, grid_hw=raw_grid_hw).unsqueeze(0)

    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    img_end_embed = model.language_model.get_input_embeddings()(
        torch.tensor([[img_end_id]], device=device)
    )
    inputs_embeds = torch.cat([vit_embeds, img_end_embed], dim=1)

    n_img_tokens = vit_embeds.shape[1]
    tgt_len = n_img_tokens + 1
    abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(merged_grid_hw, device=device)

    t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    t_indexes[:n_img_tokens] = t_idx + 1
    t_indexes[n_img_tokens] = t_idx + 2
    h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    h_indexes[:n_img_tokens] = abs_pos_h.to(torch.long)
    w_indexes[:n_img_tokens] = abs_pos_w.to(torch.long)
    indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
    return inputs_embeds, indexes, t_idx + 2


def build_packed_interleave_plan(
    model: Any,
    tokenizer: Any,
    *,
    query: str,
    all_tokens: List[int],
    text_boundaries: List[Tuple[int, int]],
    generated_images: List[Tensor],
    pixel_values: Optional[Tensor],
    grid_hw: Optional[Tensor],
    device: torch.device,
) -> PackedInterleavePlan:
    """Build one block-causal packed query/text/image-understanding sequence."""
    from .vendor.modeling_qwen3 import create_block_causal_mask

    if pixel_values is not None:
        prefix_embeds, prefix_indexes, _ = build_it2i_inputs(
            model, tokenizer, query, pixel_values, grid_hw
        )
    else:
        prefix_ids, prefix_indexes, _ = build_text_inputs(model, tokenizer, query)
        prefix_embeds = model.language_model.get_input_embeddings()(prefix_ids)

    boundaries = list(text_boundaries) if text_boundaries else [(0, len(all_tokens))]
    if boundaries:
        covered = [tok for start, end in boundaries for tok in all_tokens[start:end]]
        if covered != list(all_tokens):
            raise ValueError(
                "Packed interleave boundaries must partition all response tokens in order: "
                f"tokens={len(all_tokens)} boundaries={boundaries}"
            )
    if len(generated_images) > len(boundaries):
        raise ValueError(
            f"Packed interleave has {len(generated_images)} images but only {len(boundaries)} text segments."
        )

    embed_parts: List[Tensor] = [prefix_embeds]
    index_parts: List[Tensor] = [prefix_indexes]
    prediction_positions: List[int] = []
    text_targets: List[int] = []
    boundary_seq_lens: List[int] = []
    boundary_t_idxs: List[int] = []
    image_shapes: List[Tuple[int, int]] = []

    packed_len = int(prefix_embeds.shape[1])
    t_idx = int(prefix_indexes[0].max().item())
    predictor_pos = packed_len - 1

    for seg_i, (seg_start, seg_end) in enumerate(boundaries):
        seg_tokens = list(all_tokens[seg_start:seg_end])
        if seg_tokens:
            seg_ids = torch.tensor(seg_tokens, dtype=torch.long, device=device)
            seg_embeds = model.language_model.get_input_embeddings()(seg_ids.unsqueeze(0))
            n_seg = len(seg_tokens)
            seg_t = torch.arange(t_idx + 1, t_idx + 1 + n_seg, dtype=torch.long, device=device)
            seg_indexes = torch.stack([seg_t, torch.zeros_like(seg_t), torch.zeros_like(seg_t)], dim=0)

            for offset, target in enumerate(seg_tokens):
                prediction_positions.append(predictor_pos)
                text_targets.append(int(target))
                predictor_pos = packed_len + offset

            embed_parts.append(seg_embeds)
            index_parts.append(seg_indexes)
            packed_len += n_seg
            t_idx += n_seg

        if seg_i < len(generated_images):
            boundary_seq_lens.append(packed_len)
            boundary_t_idxs.append(t_idx)
            image = generated_images[seg_i].to(device)
            image_shapes.append((int(image.shape[1]), int(image.shape[2])))
            image_embeds, image_indexes, t_idx = build_reencoded_image_inputs(
                model, tokenizer, t_idx, image, device=device
            )
            embed_parts.append(image_embeds)
            index_parts.append(image_indexes)
            predictor_pos = packed_len + int(image_embeds.shape[1]) - 1
            packed_len += int(image_embeds.shape[1])

    inputs_embeds = torch.cat(embed_parts, dim=1)
    indexes = torch.cat(index_parts, dim=1)
    attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    return PackedInterleavePlan(
        inputs_embeds=inputs_embeds,
        indexes=indexes,
        attention_mask=attention_mask,
        prediction_positions=torch.tensor(prediction_positions, dtype=torch.long, device=device),
        text_targets=torch.tensor(text_targets, dtype=torch.long, device=device),
        image_boundary_seq_lens=boundary_seq_lens,
        image_boundary_t_idxs=boundary_t_idxs,
        image_shapes=image_shapes,
    )


def packed_context_forward(model: Any, plan: PackedInterleavePlan) -> Tuple[Tensor, FunctionalKVCache]:
    """Run one understanding forward with functional, checkpoint-safe K/V outputs."""
    hidden, layer_keys, layer_values = model.language_model.model(
        inputs_embeds=plan.inputs_embeds,
        indexes=plan.indexes,
        attention_mask=plan.attention_mask,
        image_gen_indicators=torch.zeros(
            plan.inputs_embeds.shape[:2], dtype=torch.bool, device=plan.inputs_embeds.device
        ),
        use_cache=False,
        return_current_kv=True,
    )
    cache = FunctionalKVCache(
        layers=[FunctionalKVLayer(keys=k, values=v) for k, v in zip(layer_keys, layer_values)]
    )
    return hidden, cache


def packed_logps_from_hidden(
    model: Any,
    plan: PackedInterleavePlan,
    hidden_states: Tensor,
    *,
    temperature: float,
    logprob_chunk: int = 512,
    entropy_accumulator: Optional[PolicyEntropyAccumulator] = None,
) -> Tensor:
    hidden = hidden_states[0].index_select(0, plan.prediction_positions)
    lm_head = model.language_model.lm_head
    temp = float(temperature) if float(temperature) > 0.0 else 1.0

    def _chunk_logp(
        chunk_hidden: Tensor, targets: Tensor, *, record_entropy: bool
    ) -> Tensor:
        logits = lm_head(chunk_hidden).float() / temp
        if record_entropy and entropy_accumulator is not None:
            entropy_accumulator.add_logits(logits)
        return (
            logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            - torch.logsumexp(logits, dim=1)
        )

    def _checkpointed_chunk():
        # Non-reentrant checkpoint calls this closure again during backward.
        # Record entropy only on the original forward to avoid paying twice.
        recorded = False

        def _forward(chunk_hidden: Tensor, targets: Tensor) -> Tensor:
            nonlocal recorded
            result = _chunk_logp(
                chunk_hidden, targets, record_entropy=not recorded
            )
            recorded = True
            return result

        return _forward

    logps: List[Tensor] = []
    for start in range(0, hidden.shape[0], logprob_chunk):
        end = min(start + logprob_chunk, hidden.shape[0])
        chunk_hidden = hidden[start:end]
        targets = plan.text_targets[start:end]
        if torch.is_grad_enabled() and chunk_hidden.requires_grad:
            logps.append(
                checkpoint(
                    _checkpointed_chunk(), chunk_hidden, targets, use_reentrant=False
                )
            )
        else:
            logps.append(
                _chunk_logp(chunk_hidden, targets, record_entropy=True)
            )
    return torch.cat(logps, dim=0) if logps else hidden.new_zeros((0,), dtype=torch.float32)


def packed_interleave_forward(
    model: Any,
    plan: PackedInterleavePlan,
    *,
    temperature: float,
    logprob_chunk: int = 4096,
    entropy_accumulator: Optional[PolicyEntropyAccumulator] = None,
) -> Tuple[Tensor, Any]:
    """Run one grad-capable understanding forward and return text logps + full KV."""
    hidden, full_cache = packed_context_forward(model, plan)
    logps = packed_logps_from_hidden(
        model,
        plan,
        hidden,
        temperature=temperature,
        logprob_chunk=logprob_chunk,
        entropy_accumulator=entropy_accumulator,
    )
    return logps, full_cache


def boundary_cache_view(
    full_cache: FunctionalKVCache, seq_len: int, *, config: Any = None
) -> FunctionalKVCache:
    """Return gradient-preserving prefix views of functional per-layer K/V tensors."""
    del config  # retained for call-site compatibility with the first packed implementation
    return FunctionalKVCache(
        layers=[
            FunctionalKVLayer(
                keys=layer.keys[:, :, :seq_len, :],
                values=layer.values[:, :, :seq_len, :],
            )
            for layer in full_cache.layers
        ]
    )


def score_packed_interleaved_response(
    model: Any,
    tokenizer: Any,
    *,
    query: str,
    all_tokens: List[int],
    text_boundaries: List[Tuple[int, int]],
    generated_images: List[Tensor],
    temperature: float,
    pixel_values: Optional[Tensor],
    grid_hw: Optional[Tensor],
    device: torch.device,
    entropy_accumulator: Optional[PolicyEntropyAccumulator] = None,
) -> Tensor:
    """Single-forward teacher-forced replay for text/image interleaved responses."""
    plan = build_packed_interleave_plan(
        model,
        tokenizer,
        query=query,
        all_tokens=all_tokens,
        text_boundaries=text_boundaries,
        generated_images=generated_images,
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        device=device,
    )
    logps, _ = packed_interleave_forward(
        model, plan, temperature=temperature, entropy_accumulator=entropy_accumulator
    )
    return logps


# ---------------------------------------------------------------------------
# Interleave: combined AR + Diffusion decode
# ---------------------------------------------------------------------------



def _cache_tensors(cache: Any) -> List[Tuple[Tensor, Tensor]]:
    """Return initialized K/V tensors from a rollout DynamicCache."""
    tensors: List[Tuple[Tensor, Tensor]] = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise TypeError("GeoWeave rollout batching requires initialized DynamicCache layers")
        tensors.append((keys, values))
    return tensors


def clone_select_kv_cache(cache: Any, indices: Tensor) -> Any:
    """Materialize selected batch rows as an independent DynamicCache."""
    from transformers.cache_utils import DynamicCache

    selected = [
        (keys.index_select(0, indices).clone(), values.index_select(0, indices).clone())
        for keys, values in _cache_tensors(cache)
    ]
    return DynamicCache(ddp_cache_data=selected)


def repeat_kv_cache(cache: Any, repeats: int) -> Any:
    """Materialize one prefix cache as ``repeats`` independent batch rows."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    device = _cache_tensors(cache)[0][0].device
    return clone_select_kv_cache(
        cache, torch.zeros(repeats, dtype=torch.long, device=device)
    )


def append_images_to_cache(
    model: Any,
    tokenizer: Any,
    cache: Any,
    t_idx: int,
    image_tensors: Tensor,
    *,
    device: torch.device,
) -> Tuple[Any, int, Tensor]:
    """Batch understanding-ViT re-encode for equal-shaped generated images."""
    from .vendor.modeling_neo_chat import build_abs_positions_from_grid_hw

    if image_tensors.dim() != 4:
        raise ValueError(
            f"image_tensors must be [B,3,H,W], got {tuple(image_tensors.shape)}"
        )
    batch_size = image_tensors.shape[0]
    pred_img = image_tensors.to(device=device, dtype=torch.bfloat16)
    raw_img = pred_img * 0.5 + 0.5
    img_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=torch.bfloat16
    ).view(1, 3, 1, 1)
    img_std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=torch.bfloat16
    ).view(1, 3, 1, 1)
    und_img = (raw_img - img_mean) / img_std

    ps = model.patch_size
    merge_size = int(1 / model.downsample_ratio)
    _, channels, height, width = und_img.shape
    patch_h, patch_w = height // ps, width // ps
    flatten_pv = und_img.view(
        batch_size, channels, patch_h, ps, patch_w, ps
    ).permute(0, 2, 4, 1, 3, 5)
    flatten_pv = flatten_pv.reshape(
        batch_size * patch_h * patch_w, channels * ps * ps
    )
    raw_grid_hw = torch.tensor(
        [[patch_h, patch_w]] * batch_size, device=device, dtype=torch.long
    )
    merged_grid_hw = torch.tensor(
        [[patch_h // merge_size, patch_w // merge_size]],
        device=device,
        dtype=torch.long,
    )
    vit_embeds_flat = model.extract_feature(flatten_pv, grid_hw=raw_grid_hw)
    vit_embeds = vit_embeds_flat.reshape(
        batch_size, -1, vit_embeds_flat.shape[-1]
    )

    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    img_end_ids = torch.full(
        (batch_size, 1), img_end_id, dtype=torch.long, device=device
    )
    img_end_embed = model.language_model.get_input_embeddings()(img_end_ids)
    inputs_embeds = torch.cat([vit_embeds, img_end_embed], dim=1)

    n_img_tokens = int(vit_embeds.shape[1])
    tgt_len = n_img_tokens + 1
    abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
        merged_grid_hw, device=device
    )
    t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    t_indexes[:n_img_tokens] = t_idx + 1
    t_indexes[n_img_tokens] = t_idx + 2
    h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
    h_indexes[:n_img_tokens] = abs_pos_h.to(torch.long)
    w_indexes[:n_img_tokens] = abs_pos_w.to(torch.long)
    indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    past_len = cache.get_seq_length()
    mask = torch.zeros(
        batch_size, 1, tgt_len, past_len + tgt_len, device=device
    )
    mask[:, 0, :n_img_tokens, past_len + n_img_tokens] = float("-inf")
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        indexes=indexes,
        attention_mask={"full_attention": mask},
        past_key_values=cache,
        use_cache=True,
    )
    return outputs.past_key_values, t_idx + 2, outputs.logits


def append_image_to_cache(
    model: Any,
    tokenizer: Any,
    cache: Any,
    t_idx: int,
    image_tensor: Tensor,
    *,
    device: torch.device,
) -> Tuple[Any, int, Tensor]:
    """Re-encode one generated image and append its understanding tokens to KV."""
    inputs_embeds_img, indexes, new_t_idx = build_reencoded_image_inputs(
        model, tokenizer, t_idx, image_tensor, device=device
    )
    n_img_tokens = int(inputs_embeds_img.shape[1]) - 1
    tgt_len = int(inputs_embeds_img.shape[1])
    past_len = cache.get_seq_length()
    mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
    mask[0, 0, :n_img_tokens, past_len + n_img_tokens] = float("-inf")
    outputs = model.language_model(
        inputs_embeds=inputs_embeds_img,
        indexes=indexes,
        attention_mask={"full_attention": mask},
        past_key_values=cache,
        use_cache=True,
    )
    return outputs.past_key_values, new_t_idx, outputs.logits


def append_text_tokens_to_cache(
    model: Any,
    cache: Any,
    t_idx: int,
    input_ids: Tensor,
) -> int:
    """Append text tokens to KV cache. Wraps ``model._append_text_tokens_to_cache``."""
    return model._append_text_tokens_to_cache(cache, t_idx, input_ids)


def interleave_decode(
    model: Any,
    tokenizer: Any,
    past_key_values: Any,
    t_idx: int,
    *,
    text_uncond_cache: Any | None = None,
    text_uncond_t_idx: int | None = None,
    start_logits: Tensor,
    sample_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
    max_new_tokens: int,
    max_images: int,
    stop_ids: List[int],
    img_start_token_id: int,
    diffuse_fn: Callable[..., Tuple[Tensor, Any]],
    image_size: Tuple[int, int],
    device: torch.device,
    rollout_metrics: Optional[Any] = None,
) -> Tuple[List[int], List[float], List[Tensor], List[Tuple[int, int]], Any, int]:
    """Interleaved AR + Diffusion decoding loop.

    Alternates between text generation and image generation:
    - Generates text tokens with log-probs until ``<img>`` or EOS
    - On ``<img>``: calls ``diffuse_fn`` to generate an image
    - Re-encodes the image through understanding ViT → appends to KV cache
    - Continues text generation

    Args:
        model: NEOChatModel.
        tokenizer: tokenizer.
        past_key_values: KV cache from prefix forward.
        t_idx: current temporal index.
        start_logits: logits from prefix forward ``[1, 1, V]`` or ``[1, V]``.
        sample_fn: ``(logits [1, V]) -> (token_id [1], log_prob [1])``.
        max_new_tokens: max total text tokens to generate.
        max_images: max images to generate.
        stop_ids: stop token IDs (EOS etc., NOT img_start).
        img_start_token_id: the ``<img>`` token ID.
        diffuse_fn: receives conditional/unconditional KV caches and temporal
            indexes plus ``image_size``. Called when ``<img>`` is emitted.
        image_size: ``(H, W)`` for generated images.
        device: torch device.

    Returns:
        ``(all_tokens, all_logps, generated_images, text_boundaries, past_key_values, t_idx)``
        - all_tokens: flat list of ALL text tokens across all segments
        - all_logps: corresponding log-probs
        - generated_images: list of ``[3, H, W]`` tensors
        - text_boundaries: list of ``(start, end)`` tuples for each text segment
    """
    all_tokens: List[int] = []
    all_logps: List[float] = []
    generated_images: List[Tensor] = []
    text_boundaries: List[Tuple[int, int]] = []
    n_images = 0
    total_text_tokens = 0

    # First token from prefix logits
    logits_2d = start_logits[:, -1, :] if start_logits.dim() == 3 else start_logits
    token_id, log_prob = sample_fn(logits_2d)
    next_token = token_id.view(1)

    while total_text_tokens < max_new_tokens:
        # --- Text generation segment ---
        seg_start = len(all_tokens)
        seg_tokens: List[int] = []

        while total_text_tokens < max_new_tokens:
            tid = next_token.item()

            if tid in stop_ids:
                # EOS: record and break out of everything
                all_tokens.append(tid)
                all_logps.append(log_prob.item())
                seg_tokens.append(tid)
                total_text_tokens += 1
                text_boundaries.append((seg_start, len(all_tokens)))
                if rollout_metrics is not None:
                    rollout_metrics.stop_reason = "eos"
                return (
                    all_tokens, all_logps, generated_images,
                    text_boundaries, past_key_values, t_idx,
                )

            if tid == img_start_token_id:
                if n_images >= max_images:
                    # Image limit reached: terminate trajectory, don't record
                    # this <img> so token count matches image count.
                    text_boundaries.append((seg_start, len(all_tokens)))
                    if rollout_metrics is not None:
                        rollout_metrics.stop_reason = "max_images"
                    return (
                        all_tokens, all_logps, generated_images,
                        text_boundaries, past_key_values, t_idx,
                    )
                # <img> token: record it, then switch to diffusion
                all_tokens.append(tid)
                all_logps.append(log_prob.item())
                seg_tokens.append(tid)
                total_text_tokens += 1
                break

            # Regular text token
            all_tokens.append(tid)
            all_logps.append(log_prob.item())
            seg_tokens.append(tid)
            total_text_tokens += 1

            # Forward one token to grow KV cache
            model.language_model.model.current_index = t_idx
            outputs = model.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            t_idx += 1

            token_id, log_prob = sample_fn(outputs.logits[:, -1, :])
            next_token = token_id.view(1)

        # Record text segment boundary
        if seg_tokens:
            text_boundaries.append((seg_start, len(all_tokens)))

        # Check if we stopped for <img> or exhausted tokens
        if not seg_tokens or seg_tokens[-1] != img_start_token_id:
            break

        # --- Append <img> token to KV cache ---
        model.language_model.model.current_index = t_idx
        img_start_tensor = torch.tensor([img_start_token_id], device=device)
        outputs = model.language_model(
            input_ids=img_start_tensor.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        t_idx += 1

        if text_uncond_cache is not None:
            if text_uncond_t_idx is None:
                raise RuntimeError("text CFG cache is missing its temporal index")
            model.language_model.model.current_index = text_uncond_t_idx
            uncond_outputs = model.language_model(
                input_ids=img_start_tensor.unsqueeze(0),
                past_key_values=text_uncond_cache,
                use_cache=True,
            )
            text_uncond_cache = uncond_outputs.past_key_values
            text_uncond_t_idx += 1

        # --- Diffusion: generate image ---
        image_tensor, _ = diffuse_fn(
            past_key_values,
            t_idx,
            text_uncond_cache,
            text_uncond_t_idx,
            image_size,
        )
        generated_images.append(image_tensor.detach().cpu())
        n_images += 1

        # --- Re-encode image through understanding ViT → append to KV cache ---
        with torch.no_grad():
            cache_len_before = int(past_key_values.get_seq_length())
            past_key_values, t_idx, reenc_logits = append_image_to_cache(
                model, tokenizer, past_key_values, t_idx,
                image_tensor, device=device,
            )
            if rollout_metrics is not None:
                rollout_metrics.image_context_tokens += (
                    int(past_key_values.get_seq_length()) - cache_len_before
                )
            if text_uncond_cache is not None:
                if text_uncond_t_idx is None:
                    raise RuntimeError("text CFG cache is missing its temporal index")
                text_uncond_cache, text_uncond_t_idx, _ = append_image_to_cache(
                    model,
                    tokenizer,
                    text_uncond_cache,
                    text_uncond_t_idx,
                    image_tensor,
                    device=device,
                )

        # Sample next token from the logits after image re-encoding
        token_id, log_prob = sample_fn(reenc_logits[:, -1, :])
        next_token = token_id.view(1)

    if rollout_metrics is not None:
        rollout_metrics.stop_reason = "max_new_tokens"

    return (
        all_tokens, all_logps, generated_images,
        text_boundaries, past_key_values, t_idx,
    )


# ---------------------------------------------------------------------------
# Interleave: segmented replay scoring
# ---------------------------------------------------------------------------


def score_interleaved_response(
    model: Any,
    tokenizer: Any,
    *,
    query: str,
    all_tokens: List[int],
    text_boundaries: List[Tuple[int, int]],
    generated_images: List[Tensor],
    temperature: float = 1.0,
    logprob_chunk: int = 4096,
    pixel_values: Optional[Tensor] = None,
    grid_hw: Optional[Tensor] = None,
    device: torch.device,
) -> Tensor:
    """Segmented teacher-forced replay for interleaved text+image sequences.

    For each text segment:
    1. If preceded by an image: re-encode through understanding ViT (no_grad) → append to KV cache
    2. Teacher-forced forward on the text segment (with grad) → gather new_logp

    Args:
        model: NEOChatModel.
        tokenizer: tokenizer.
        query: the original prompt query string.
        all_tokens: flat list of all text tokens from rollout.
        text_boundaries: list of (start, end) tuples for each text segment.
        generated_images: list of image tensors [3,H,W] from rollout.
        temperature: for log-prob scaling.
        logprob_chunk: chunk size for lm_head.
        pixel_values: optional input images for the prompt.
        grid_hw: optional grid HW for input images.
        device: torch device.

    Returns:
        ``[total_text_tokens]`` per-token log-probs in fp32.
    """
    # Build initial context from query
    if pixel_values is not None:
        input_embeds, indexes, attn_mask = build_it2i_inputs(
            model, tokenizer, query, pixel_values, grid_hw
        )
        with torch.no_grad():
            past_kv, hidden = prefix_forward_embeds(
                model, input_embeds, indexes, attn_mask
            )
        t_idx = indexes[0].max().item()
    else:
        input_ids, indexes, attn_mask = build_text_inputs(model, tokenizer, query)
        with torch.no_grad():
            past_kv, hidden = prefix_forward(model, input_ids, indexes, attn_mask)
        t_idx = indexes[0].max().item()

    lm_head = model.language_model.lm_head
    with torch.no_grad():
        prev_logits = lm_head(hidden[0, -1:])  # [1, vocab] predicts first response token

    all_logps: List[Tensor] = []
    img_idx = 0

    for seg_i, (seg_start, seg_end) in enumerate(text_boundaries):
        seg_token_ids = all_tokens[seg_start:seg_end]
        if not seg_token_ids:
            continue

        n_seg = len(seg_token_ids)
        seg_ids = torch.tensor(seg_token_ids, dtype=torch.long, device=device)

        # Score first token from prior context (prefix hidden or post-image logits)
        first_logits = prev_logits.float() / temperature
        lp_first = (
            first_logits.gather(1, seg_ids[0:1].unsqueeze(0)).squeeze()
            - torch.logsumexp(first_logits, dim=-1).squeeze()
        )
        all_logps.append(lp_first.detach().unsqueeze(0))

        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + n_seg, dtype=torch.long, device=device)
        h_indexes = torch.zeros(n_seg, dtype=torch.long, device=device)
        w_indexes = torch.zeros(n_seg, dtype=torch.long, device=device)
        seg_indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

        # Causal mask attending to prefix + causal within segment
        past_len = past_kv.get_seq_length()
        mask = torch.zeros(1, 1, n_seg, past_len + n_seg, device=device)
        causal = torch.tril(torch.ones(n_seg, n_seg, device=device))
        causal = torch.where(causal == 1, 0.0, float("-inf"))
        mask[:, :, :, past_len:] = causal
        attn_mask_seg = {"full_attention": mask}

        # Forward through LLM und branch with gradients
        seg_input_ids = seg_ids.unsqueeze(0)  # [1, n_seg]
        outputs = model.language_model.model(
            input_ids=seg_input_ids,
            indexes=seg_indexes,
            attention_mask=attn_mask_seg,
            past_key_values=past_kv,
            use_cache=False,
        )

        # hidden[i] predicts seg_ids[i+1]; score seg_ids[1:] from hidden[:-1]
        hidden_out = outputs.last_hidden_state[0]  # [n_seg, C]

        # Score tokens seg_ids[1:n_seg] from hidden[0:n_seg-1]
        if n_seg > 1:
            scoring_hidden = hidden_out[:-1]  # [n_seg-1, C]
            scoring_targets = seg_ids[1:]  # [n_seg-1]

            seg_logps = []
            for ci in range(0, n_seg - 1, logprob_chunk):
                ce = min(ci + logprob_chunk, n_seg - 1)
                ch = scoring_hidden[ci:ce]
                if ch.requires_grad:
                    logits = torch.utils.checkpoint.checkpoint(
                        lm_head, ch, use_reentrant=False
                    )
                else:
                    logits = lm_head(ch)
                logits = logits.float() / temperature
                tgt = scoring_targets[ci:ce]
                lp = logits.gather(1, tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(logits, dim=1)
                seg_logps.append(lp)

            all_logps.extend(seg_logps)

        # Update KV cache for next segment (under no_grad)
        with torch.no_grad():
            t_idx = append_text_tokens_to_cache(
                model, past_kv, t_idx, seg_ids.unsqueeze(0)
            )

        # If there's a generated image after this segment, re-encode it
        if img_idx < len(generated_images):
            with torch.no_grad():
                past_kv, t_idx, reenc_logits = append_image_to_cache(
                    model, tokenizer, past_kv, t_idx,
                    generated_images[img_idx].to(device),
                    device=device,
                )
                prev_logits = reenc_logits[:, -1, :]  # [1, vocab] predicts next segment's first token
            img_idx += 1

    if not all_logps:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.cat(all_logps, dim=0)


# ---------------------------------------------------------------------------
# Inline MSE: merged forward scoring with velocity MSE
# ---------------------------------------------------------------------------


def _mse_target_steps(latent_segment: Any, mse_steps: int) -> List[int]:
    if latent_segment is None or latent_segment.sigmas is None:
        return []
    if latent_segment.sde_indices is not None:
        return [int(x) for x in latent_segment.sde_indices.tolist()[:mse_steps]]
    return list(range(min(int(mse_steps), max(0, len(latent_segment.sigmas) - 1))))


@dataclass
class PackedMSEJob:
    boundary_index: int
    latent_segment: Any
    step_index: int
    is_real: bool


def build_packed_mse_jobs(
    plan: PackedInterleavePlan,
    latent_segments: List[Any],
    *,
    mse_steps: int,
    target_job_count: int,
) -> List[PackedMSEJob]:
    jobs: List[PackedMSEJob] = []
    for image_idx, lat_seg in enumerate(latent_segments[: len(plan.image_boundary_seq_lens)]):
        for step_idx in _mse_target_steps(lat_seg, mse_steps):
            jobs.append(PackedMSEJob(image_idx, lat_seg, step_idx, True))
    if len(jobs) > target_job_count:
        raise ValueError(f"local MSE jobs {len(jobs)} exceed DP target {target_job_count}")
    jobs.extend(
        PackedMSEJob(-1, None, -1, False)
        for _ in range(target_job_count - len(jobs))
    )
    return jobs


def _packed_mse_job_inputs(
    model: Any,
    plan: PackedInterleavePlan,
    job: PackedMSEJob,
    cache: Any,
    *,
    diffusion_stage: Any,
    diffusion_params: Any,
    device: torch.device,
) -> Tuple[Dict[str, Any], Tensor, Tensor]:
    if job.is_real:
        image_idx = job.boundary_index
        seq_len = plan.image_boundary_seq_lens[image_idx]
        t_idx = plan.image_boundary_t_idxs[image_idx]
        image_shape = plan.image_shapes[image_idx]
        boundary = boundary_cache_view(cache, seq_len, config=model.language_model.config)
        schedule = job.latent_segment.sigmas.to(device)
        x_t = job.latent_segment.latents_at(job.step_index)[0].to(device)
        sigma = schedule[job.step_index]
    else:
        seq_len = int(plan.inputs_embeds.shape[1])
        t_idx = int(plan.indexes[0].max().item())
        image_shape = (
            int(getattr(diffusion_params, "height", 256)),
            int(getattr(diffusion_params, "width", 256)),
        )
        boundary = boundary_cache_view(cache, seq_len, config=model.language_model.config)
        h, w = image_shape
        pm = int(model.patch_size * int(1 / model.downsample_ratio))
        x_t = torch.zeros(
            (h // pm) * (w // pm), pm * pm * 3,
            device=device,
            dtype=torch.float32,
        )
        sigma = torch.tensor(0.5, device=device, dtype=torch.float32)

    fwd_kwargs = diffusion_stage.build_forward_kwargs_from_kv(
        boundary, t_idx, image_shape, params=diffusion_params, device=device
    )
    return fwd_kwargs, x_t, sigma


def packed_reference_logps_and_velocities(
    model: Any,
    plan: PackedInterleavePlan,
    jobs: List[PackedMSEJob],
    *,
    diffusion_stage: Any,
    diffusion_params: Any,
    temperature: float,
    compute_logps: bool,
    device: torch.device,
) -> Tuple[Optional[Tensor], List[Tensor]]:
    """One reference context forward for optional AR logps and MSE velocities.

    The context backbone is intentionally shared: enabling AR reference KL on
    the inline-MSE path adds only the reference LM-head scoring, not another
    transformer forward.
    """
    ref_hidden, ref_cache = packed_context_forward(model, plan)
    ref_logps = (
        packed_logps_from_hidden(
            model, plan, ref_hidden, temperature=temperature
        ).detach()
        if compute_logps
        else None
    )
    refs: List[Tensor] = []
    for job in jobs:
        fwd_kwargs, x_t, sigma = _packed_mse_job_inputs(
            model, plan, job, ref_cache,
            diffusion_stage=diffusion_stage,
            diffusion_params=diffusion_params,
            device=device,
        )
        refs.append(
            diffusion_stage.predict_velocity_at(
                fwd_kwargs, sample=x_t, sigma=sigma, params=diffusion_params
            ).detach()
        )
    return ref_logps, refs


def packed_policy_logps_and_mse(
    model: Any,
    plan: PackedInterleavePlan,
    jobs: List[PackedMSEJob],
    ref_velocities: List[Tensor],
    *,
    diffusion_stage: Any,
    diffusion_params: Any,
    temperature: float,
    device: torch.device,
    entropy_accumulator: Optional[PolicyEntropyAccumulator] = None,
) -> Tuple[Tensor, Tensor, int]:
    policy_hidden, policy_cache = packed_context_forward(model, plan)
    logps = packed_logps_from_hidden(
        model,
        plan,
        policy_hidden,
        temperature=temperature,
        entropy_accumulator=entropy_accumulator,
    )
    mse_sum = logps.new_zeros(())
    real_count = 0
    for job, v_ref in zip(jobs, ref_velocities):
        fwd_kwargs, x_t, sigma = _packed_mse_job_inputs(
            model, plan, job, policy_cache,
            diffusion_stage=diffusion_stage,
            diffusion_params=diffusion_params,
            device=device,
        )
        v_policy = diffusion_stage.predict_velocity_at(
            fwd_kwargs, sample=x_t, sigma=sigma, params=diffusion_params
        )
        if job.is_real:
            mse_sum = mse_sum + ((v_policy.float() - v_ref.float()) ** 2).mean()
            real_count += 1
        else:
            mse_sum = mse_sum + v_policy.sum() * 0.0
    return logps, mse_sum, real_count


__all__ = [
    "build_query",
    "prepare_input_image",
    "insert_image_tokens",
    "build_text_inputs",
    "build_it2i_inputs",
    "prefix_forward",
    "prefix_forward_embeds",
    "build_image_embeds",
    "build_image_indexes",
    "predict_v",
    "patchify",
    "unpatchify",
    "apply_time_schedule",
    "euler_step",
    "decode_text",
    "score_response",
    "PolicyEntropyAccumulator",
    "clone_select_kv_cache",
    "repeat_kv_cache",
    "append_images_to_cache",
    "append_image_to_cache",
    "append_text_tokens_to_cache",
    "interleave_decode",
    "score_interleaved_response",
    "FunctionalKVLayer",
    "FunctionalKVCache",
    "build_packed_interleave_plan",
    "packed_context_forward",
    "packed_logps_from_hidden",
    "boundary_cache_view",
    "score_packed_interleaved_response",
    "PackedMSEJob",
    "build_packed_mse_jobs",
    "packed_reference_logps_and_velocities",
    "packed_policy_logps_and_mse",
]
