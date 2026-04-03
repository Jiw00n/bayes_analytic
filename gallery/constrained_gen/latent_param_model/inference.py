from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .adapter import GeneratorRegistry
from .dataset import collate_prepared_samples
from .tokenizer import ParamTokenizer


@dataclass
class DecodeResult:
    predicted_param_dict: Dict[str, int]
    predicted_token_ids: List[int]
    predicted_tokens: List[str]


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


@torch.no_grad()
def greedy_decode_sample(
    model,
    sample,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
) -> DecodeResult:
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

    decoder_input_ids: List[int] = [tokenizer.bos_id]
    decoded_values: List[int] = []
    decoded_token_ids: List[int] = []

    for step_idx, var_name in enumerate(ordered_names):
        candidate_values = oracle.candidate_values(var_name)
        mask = tokenizer.candidate_mask_from_values(var_name, candidate_values, device=device)
        if not mask.any():
            mask = tokenizer.pad_only_mask(device=device)

        step_input = torch.tensor([decoder_input_ids], dtype=torch.long, device=device)
        step_var_ids = torch.tensor(
            [[tokenizer.var_to_id[name] for name in ordered_names[: len(decoder_input_ids)]]],
            dtype=torch.long,
            device=device,
        )

        logits = model.decode(
            step_input,
            step_var_ids,
            memory,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[0, -1].masked_fill(~mask, float("-inf"))
        pred_token_id = int(torch.argmax(step_logits).item())
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
) -> List[DecodeResult]:
    if not samples:
        return []

    model.eval()
    batch = collate_prepared_samples(samples, tokenizer)
    enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
    enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, _, memory = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)

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
    ordered_var_ids = [list(sample.encoder_var_ids) for sample in samples]
    max_steps = max(len(names) for names in ordered_names)
    batch_size = len(samples)
    decoder_input_ids = torch.full(
        (batch_size, max_steps + 1),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    decoder_var_ids = torch.full(
        (batch_size, max_steps + 1),
        tokenizer.var_pad_id,
        dtype=torch.long,
        device=device,
    )
    decoder_input_ids[:, 0] = tokenizer.bos_id
    current_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    for sample_idx, var_ids in enumerate(ordered_var_ids):
        if var_ids:
            decoder_var_ids[sample_idx, 0] = int(var_ids[0])

    decoded_values: List[List[int]] = [[] for _ in samples]
    decoded_token_ids: List[List[int]] = [[] for _ in samples]

    for step_idx in range(max_steps):
        current_width = int(current_lengths.max().item())
        step_input = decoder_input_ids[:, :current_width]
        step_var = decoder_var_ids[:, :current_width]

        logits = model.decode(
            step_input,
            step_var,
            memory,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[:, current_width - 1, :]
        step_masks = torch.zeros_like(step_logits, dtype=torch.bool)

        active_indices: List[int] = []
        candidate_lists: List[List[int]] = []
        for sample_idx, sample_names in enumerate(ordered_names):
            if step_idx >= len(sample_names):
                continue
            active_indices.append(sample_idx)
            var_name = sample_names[step_idx]
            candidate_values = oracles[sample_idx].candidate_values(var_name)
            candidate_lists.append(list(candidate_values))
            step_masks[sample_idx] = tokenizer.candidate_mask_from_values(
                var_name,
                candidate_values,
                device=device,
            )
            if not step_masks[sample_idx].any():
                step_masks[sample_idx] = tokenizer.pad_only_mask(device=device)

        if not active_indices:
            continue

        masked_logits = step_logits.masked_fill(~step_masks, float("-inf"))
        pred_token_ids = torch.argmax(masked_logits[active_indices], dim=-1).tolist()

        for local_idx, sample_idx in enumerate(active_indices):
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
