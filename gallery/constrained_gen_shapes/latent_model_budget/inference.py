from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .adapter import GeneratorRegistry
from .dataset import collate_prepared_samples
from .tokenizer import ParamTokenizer


@dataclass
class DecodeResult:
    predicted_param_dict: Dict[str, int]
    predicted_token_ids: List[int]
    predicted_tokens: List[str]


@dataclass
class SamplingOptions:
    strategy: str = "greedy"  # "greedy" | "sampling"
    temperature: float = 1.0
    top_k: int = 0            # 0 disables top-k
    top_p: float = 1.0        # 1.0 disables top-p
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, sampling_cfg) -> "SamplingOptions":
        if sampling_cfg is None:
            return cls()
        return cls(
            strategy=str(getattr(sampling_cfg, "strategy", "greedy")),
            temperature=float(getattr(sampling_cfg, "temperature", 1.0)),
            top_k=int(getattr(sampling_cfg, "top_k", 0) or 0),
            top_p=float(getattr(sampling_cfg, "top_p", 1.0)),
            seed=getattr(sampling_cfg, "seed", None),
        )


def _sample_token_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    options: SamplingOptions,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample a single token id from 1-D ``logits`` restricted to ``mask``.

    ``logits`` and ``mask`` must be 1-D tensors of identical shape. The
    returned token id is always a legal candidate (i.e. ``mask`` is True at
    that position); if the configured truncation empties the distribution the
    fallback is argmax over the masked logits.
    """
    masked = logits.masked_fill(~mask, float("-inf"))
    strategy = (options.strategy or "greedy").lower()
    if strategy == "greedy":
        return int(torch.argmax(masked).item())

    temperature = float(options.temperature)
    if temperature <= 0.0:
        return int(torch.argmax(masked).item())
    if temperature != 1.0:
        masked = masked / temperature

    if options.top_k and options.top_k > 0:
        finite_count = int(torch.isfinite(masked).sum().item())
        k = min(int(options.top_k), max(finite_count, 1))
        topk_vals, _ = torch.topk(masked, k)
        kth = topk_vals[-1]
        masked = torch.where(
            masked < kth,
            torch.full_like(masked, float("-inf")),
            masked,
        )

    if 0.0 < options.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(masked, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > options.top_p
        # Always keep the top-1 candidate so we never empty the set.
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        masked = torch.full_like(masked, float("-inf"))
        masked.scatter_(0, sorted_indices, sorted_logits)

    finite_mask = torch.isfinite(masked)
    if not bool(finite_mask.any()):
        return int(torch.argmax(logits.masked_fill(~mask, float("-inf"))).item())

    probs = torch.softmax(masked, dim=-1)
    total = float(probs.sum().item())
    if not (total > 0.0) or not torch.isfinite(probs).all():
        return int(torch.argmax(masked).item())

    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item())


def reconstruct_param_dict(ordered_names: Sequence[str], ordered_values: Sequence[int]) -> Dict[str, int]:
    return {name: int(value) for name, value in zip(ordered_names, ordered_values)}


def _resolve_decoded_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    token_id: int,
    candidate_values: Sequence[int],
) -> int:
    token = tokenizer.id_to_token[int(token_id)]
    value = tokenizer.token_to_value(var_name, token)
    if value is not None:
        return int(value)
    if not candidate_values:
        raise RuntimeError(f"No valid candidates available for {var_name}")
    return int(candidate_values[0])


def _fixed_token_id_for_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    value: int,
) -> int:
    token = tokenizer.value_to_token(var_name, int(value))
    return int(tokenizer.token_to_id.get(token, tokenizer.unk_id))


@torch.no_grad()
def greedy_decode_sample(
    model,
    sample,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    *,
    sampling_options: Optional[SamplingOptions] = None,
    rng: Optional[torch.Generator] = None,
) -> DecodeResult:
    options = sampling_options or SamplingOptions()
    model.eval()

    enc_ids = torch.tensor([sample.encoder_token_ids], dtype=torch.long, device=device)
    enc_var_ids = torch.tensor([sample.encoder_var_ids], dtype=torch.long, device=device)

    enc_pad = enc_ids.eq(tokenizer.pad_id)
    mu, logvar, z, memory = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)

    oracle = registry.build_oracle(
        task_index=sample.task_index,
        sketch_index=sample.sketch_index,
        workload_key=sample.workload_key,
        target_kind=sample.target_kind,
    )
    ordered_names = list(sample.ordered_param_names)
    full_var_ids = list(sample.encoder_var_ids)  # [shape_var | param_var]

    decoder_input_ids: List[int] = list(sample.shape_token_ids) + [tokenizer.param_start_id]
    decoded_values: List[int] = []
    decoded_token_ids: List[int] = []

    for step_idx, var_name in enumerate(ordered_names):
        candidate_values = oracle.candidate_values(var_name)
        if len(candidate_values) == 1:
            pred_value = int(candidate_values[0])
            pred_token_id = _fixed_token_id_for_value(tokenizer, var_name, pred_value)
            oracle.assign(var_name, pred_value)
            decoded_values.append(pred_value)
            decoded_token_ids.append(pred_token_id)
            decoder_input_ids.append(pred_token_id)
            continue
        mask = tokenizer.candidate_mask_from_values(var_name, candidate_values, device=device)
        if not mask.any():
            mask = tokenizer.pad_only_mask(device=device)

        step_input = torch.tensor([decoder_input_ids], dtype=torch.long, device=device)
        step_var_ids = torch.tensor(
            [full_var_ids[: len(decoder_input_ids)]],
            dtype=torch.long,
            device=device,
        )

        logits = model.decode(
            step_input,
            step_var_ids,
            memory,
            z,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[0, -1]
        pred_token_id = _sample_token_from_logits(step_logits, mask, options, rng)
        pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

        oracle.assign(var_name, pred_value)
        decoded_values.append(pred_value)
        decoded_token_ids.append(pred_token_id)
        decoder_input_ids.append(pred_token_id)

    predicted_dict = reconstruct_param_dict(ordered_names, decoded_values)
    return DecodeResult(
        predicted_param_dict=predicted_dict,
        predicted_token_ids=decoded_token_ids,
        predicted_tokens=[tokenizer.id_to_token[idx] for idx in decoded_token_ids],
    )


@torch.no_grad()
def greedy_decode_batch(
    model,
    samples,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    *,
    sampling_options: Optional[SamplingOptions] = None,
    rng: Optional[torch.Generator] = None,
) -> List[DecodeResult]:
    if not samples:
        return []

    options = sampling_options or SamplingOptions()
    model.eval()
    batch = collate_prepared_samples(samples, tokenizer)
    enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
    enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, z, memory = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)

    oracles = [
        registry.build_oracle(
            task_index=sample.task_index,
            sketch_index=sample.sketch_index,
            workload_key=sample.workload_key,
            target_kind=sample.target_kind,
        )
        for sample in samples
    ]
    ordered_names = [list(sample.ordered_param_names) for sample in samples]
    # ``encoder_var_ids`` = [shape_var_ids | param_var_ids]; reuse it so decoder
    # var-id lookups at pos >= shape_len naturally pick the right param var.
    ordered_var_ids = [list(sample.encoder_var_ids) for sample in samples]
    shape_lens = [len(sample.shape_token_ids) for sample in samples]
    max_steps = max(len(names) for names in ordered_names)
    max_shape_len = max(shape_lens) if shape_lens else 0
    batch_size = len(samples)
    decoder_input_ids = torch.full(
        (batch_size, max_shape_len + max_steps + 1),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    decoder_var_ids = torch.full(
        (batch_size, max_shape_len + max_steps + 1),
        tokenizer.var_pad_id,
        dtype=torch.long,
        device=device,
    )
    current_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for sample_idx, sample in enumerate(samples):
        shape_len = shape_lens[sample_idx]
        if shape_len:
            decoder_input_ids[sample_idx, :shape_len] = torch.tensor(
                sample.shape_token_ids, dtype=torch.long, device=device
            )
            decoder_var_ids[sample_idx, :shape_len] = torch.tensor(
                sample.shape_var_ids, dtype=torch.long, device=device
            )
        decoder_input_ids[sample_idx, shape_len] = tokenizer.param_start_id
        # BOS position var = first param var (= ordered_var_ids[shape_len]).
        if len(ordered_var_ids[sample_idx]) > shape_len:
            decoder_var_ids[sample_idx, shape_len] = int(ordered_var_ids[sample_idx][shape_len])
        current_lengths[sample_idx] = shape_len + 1

    decoded_values: List[List[int]] = [[] for _ in samples]
    decoded_token_ids: List[List[int]] = [[] for _ in samples]

    for step_idx in range(max_steps):
        variable_indices: List[int] = []
        candidate_lists: List[List[int]] = []
        for sample_idx, sample_names in enumerate(ordered_names):
            if step_idx >= len(sample_names):
                continue
            var_name = sample_names[step_idx]
            candidate_values = oracles[sample_idx].candidate_values(var_name)
            if len(candidate_values) == 1:
                pred_value = int(candidate_values[0])
                pred_token_id = _fixed_token_id_for_value(tokenizer, var_name, pred_value)
                oracles[sample_idx].assign(var_name, pred_value)
                decoded_values[sample_idx].append(pred_value)
                decoded_token_ids[sample_idx].append(pred_token_id)

                pos = int(current_lengths[sample_idx].item())
                decoder_input_ids[sample_idx, pos] = pred_token_id
                if pos < len(ordered_var_ids[sample_idx]):
                    decoder_var_ids[sample_idx, pos] = ordered_var_ids[sample_idx][pos]
                current_lengths[sample_idx] = pos + 1
                continue

            variable_indices.append(sample_idx)
            candidate_lists.append(list(candidate_values))

        if not variable_indices:
            continue

        subset_indices = torch.tensor(variable_indices, dtype=torch.long, device=device)
        subset_lengths = current_lengths[subset_indices]
        current_width = int(subset_lengths.max().item())
        step_input = decoder_input_ids.index_select(0, subset_indices)[:, :current_width]
        step_var = decoder_var_ids.index_select(0, subset_indices)[:, :current_width]
        logits = model.decode(
            step_input,
            step_var,
            memory.index_select(0, subset_indices),
            z.index_select(0, subset_indices),
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        gather_pos = (subset_lengths - 1).to(dtype=torch.long)
        step_logits = logits[torch.arange(len(variable_indices), device=device), gather_pos, :]
        step_masks = torch.zeros_like(step_logits, dtype=torch.bool)
        for local_idx, sample_idx in enumerate(variable_indices):
            var_name = ordered_names[sample_idx][step_idx]
            step_masks[local_idx] = tokenizer.candidate_mask_from_values(
                var_name,
                candidate_lists[local_idx],
                device=device,
            )
            if not step_masks[local_idx].any():
                step_masks[local_idx] = tokenizer.pad_only_mask(device=device)

        pred_token_ids: List[int] = []
        for local_idx in range(step_logits.shape[0]):
            pred_token_ids.append(
                _sample_token_from_logits(
                    step_logits[local_idx],
                    step_masks[local_idx],
                    options,
                    rng,
                )
            )

        for local_idx, sample_idx in enumerate(variable_indices):
            var_name = ordered_names[sample_idx][step_idx]
            candidate_values = candidate_lists[local_idx]
            pred_token_id = int(pred_token_ids[local_idx])
            pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

            oracles[sample_idx].assign(var_name, pred_value)
            decoded_values[sample_idx].append(pred_value)
            decoded_token_ids[sample_idx].append(pred_token_id)

            pos = int(current_lengths[sample_idx].item())
            decoder_input_ids[sample_idx, pos] = pred_token_id
            if pos < len(ordered_var_ids[sample_idx]):
                decoder_var_ids[sample_idx, pos] = ordered_var_ids[sample_idx][pos]
            current_lengths[sample_idx] = pos + 1

    results: List[DecodeResult] = []
    for sample_idx, sample in enumerate(samples):
        predicted_dict = reconstruct_param_dict(
            sample.ordered_param_names,
            decoded_values[sample_idx],
        )
        results.append(
            DecodeResult(
                predicted_param_dict=predicted_dict,
                predicted_token_ids=decoded_token_ids[sample_idx],
                predicted_tokens=[tokenizer.id_to_token[idx] for idx in decoded_token_ids[sample_idx]],
            )
        )
    return results


def pretty_print_reconstruction(sample, decode_result: DecodeResult) -> str:
    lines = [f"sample_id={sample.sample_id}"]
    for name, gold in zip(sample.ordered_param_names, sample.ordered_param_values):
        pred = decode_result.predicted_param_dict.get(name)
        mark = "OK" if pred == gold else "DIFF"
        lines.append(f"  {name}: gold={gold} pred={pred} [{mark}]")
    return "\n".join(lines)
