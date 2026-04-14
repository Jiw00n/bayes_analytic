from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch

try:
    from latent_model_budget.adapter import GeneratorRegistry, JsonSampleRecord
    from latent_model_budget.config import build_config
    from latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from latent_model_budget.model import LatentParamVAE
    from latent_model_budget.tokenizer import ParamTokenizer
    from latent_model_budget.train_eval import (
        _batch_to_device,
        _build_singleton_position_mask,
        _compress_teacher_forcing_batch,
    )
except ImportError:  # pragma: no cover
    import sys

    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    from .latent_model_budget.adapter import GeneratorRegistry, JsonSampleRecord
    from .latent_model_budget.config import build_config
    from .latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from .latent_model_budget.model import LatentParamVAE
    from .latent_model_budget.tokenizer import ParamTokenizer
    from .latent_model_budget.train_eval import (
        _batch_to_device,
        _build_singleton_position_mask,
        _compress_teacher_forcing_batch,
    )


def _dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    return obj


def _make_model_cfg_namespace(payload: Dict[str, Any]) -> Any:
    merged = {k: v for k, v in vars(build_config().model).items()}
    merged.update(dict(payload))
    return _dict_to_namespace(merged)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _assert_same_tokenizer(saved: ParamTokenizer, built: ParamTokenizer) -> None:
    problems: List[str] = []
    if saved.id_to_token != built.id_to_token:
        problems.append("id_to_token mismatch")
    if saved.id_to_var != built.id_to_var:
        problems.append("id_to_var mismatch")
    if problems:
        raise RuntimeError(
            "Checkpoint tokenizer and rebuilt dataset tokenizer do not match: "
            + ", ".join(problems)
            + ". Check json_paths / split config / budget flag."
        )


class RunningStats:
    __slots__ = (
        "total",
        "correct",
        "wrong",
        "right_cost_sum",
        "right_cost_count",
        "wrong_cost_sum",
        "wrong_cost_count",
    )

    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.wrong = 0
        self.right_cost_sum = 0.0
        self.right_cost_count = 0
        self.wrong_cost_sum = 0.0
        self.wrong_cost_count = 0

    def update(self, *, correct: bool, cost: float | None) -> None:
        self.total += 1
        if correct:
            self.correct += 1
            if cost is not None:
                self.right_cost_sum += float(cost)
                self.right_cost_count += 1
        else:
            self.wrong += 1
            if cost is not None:
                self.wrong_cost_sum += float(cost)
                self.wrong_cost_count += 1

    def to_dict(self) -> Dict[str, Any]:
        acc = self.correct / self.total if self.total else 0.0
        right_cost_mean = (
            self.right_cost_sum / self.right_cost_count if self.right_cost_count else None
        )
        wrong_cost_mean = (
            self.wrong_cost_sum / self.wrong_cost_count if self.wrong_cost_count else None
        )
        cost_gap_if_wrong = None
        if right_cost_mean is not None and wrong_cost_mean is not None:
            cost_gap_if_wrong = float(right_cost_mean - wrong_cost_mean)
        return {
            "total": int(self.total),
            "correct": int(self.correct),
            "wrong": int(self.wrong),
            "accuracy": float(acc),
            "right_cost_mean": None if right_cost_mean is None else float(right_cost_mean),
            "wrong_cost_mean": None if wrong_cost_mean is None else float(wrong_cost_mean),
            "cost_gap_if_wrong": cost_gap_if_wrong,
        }


class SplitAnalyzer:
    def __init__(self, topk_samples: int = 30):
        self.topk_samples = int(topk_samples)
        self.var_stats: dict[str, RunningStats] = defaultdict(RunningStats)
        self.pos_stats: dict[int, RunningStats] = defaultdict(RunningStats)
        self.family_stats: dict[str, RunningStats] = defaultdict(RunningStats)
        self.sample_error_hist: Counter[int] = Counter()
        self.num_samples = 0
        self.num_exact = 0
        self.num_tokens = 0
        self.num_correct_tokens = 0
        self.sample_records: list[dict[str, Any]] = []

    @staticmethod
    def _family_of(var_name: str) -> str:
        if var_name.startswith("sp_"):
            return "sp"
        if var_name.startswith("ur_"):
            return "ur"
        if var_name.startswith("thread_budget"):
            return "thread_budget"
        if var_name.startswith("vthread_budget"):
            return "vthread_budget"
        return "other"

    def add_sample(
        self,
        *,
        sample_id: str,
        wrong_vars: Sequence[str],
        wrong_positions: Sequence[int],
        ordered_var_names: Sequence[str],
        correctness: Sequence[bool],
        cost: float | None,
        position_ids: Sequence[int] | None = None,
    ) -> None:
        self.num_samples += 1
        error_count = int(sum(1 for ok in correctness if not ok))
        self.sample_error_hist[error_count] += 1
        if error_count == 0:
            self.num_exact += 1
        self.num_tokens += len(correctness)
        self.num_correct_tokens += int(sum(1 for ok in correctness if ok))

        if position_ids is None:
            position_ids = list(range(len(correctness)))
        for pos_id, var_name, ok in zip(position_ids, ordered_var_names, correctness):
            self.var_stats[var_name].update(correct=bool(ok), cost=cost)
            self.pos_stats[int(pos_id)].update(correct=bool(ok), cost=cost)
            self.family_stats[self._family_of(var_name)].update(correct=bool(ok), cost=cost)

        if error_count > 0:
            self.sample_records.append(
                {
                    "sample_id": str(sample_id),
                    "error_count": int(error_count),
                    "wrong_positions": [int(x) for x in wrong_positions],
                    "wrong_vars": list(wrong_vars),
                    "cost": None if cost is None else float(cost),
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        token_acc = self.num_correct_tokens / self.num_tokens if self.num_tokens else 0.0
        exact = self.num_exact / self.num_samples if self.num_samples else 0.0

        non_exact = self.num_samples - self.num_exact
        one_error_only = int(self.sample_error_hist.get(1, 0))
        one_error_share_among_nonexact = (
            one_error_only / non_exact if non_exact > 0 else 0.0
        )

        sorted_var_stats = sorted(
            ((name, stats.to_dict()) for name, stats in self.var_stats.items()),
            key=lambda item: (item[1]["accuracy"], -item[1]["wrong"], item[0]),
        )
        sorted_pos_stats = sorted(
            ((str(pos), stats.to_dict()) for pos, stats in self.pos_stats.items()),
            key=lambda item: int(item[0]),
        )
        sorted_family_stats = sorted(
            ((name, stats.to_dict()) for name, stats in self.family_stats.items()),
            key=lambda item: item[0],
        )

        top_cost_sensitive = sorted(
            (
                {"var_name": name, **stats.to_dict()}
                for name, stats in self.var_stats.items()
                if stats.wrong > 0 and stats.right_cost_count > 0 and stats.wrong_cost_count > 0
            ),
            key=lambda item: (
                -(item["cost_gap_if_wrong"] if item["cost_gap_if_wrong"] is not None else float("-inf")),
                -item["wrong"],
                item["var_name"],
            ),
        )

        hardest_samples = sorted(
            self.sample_records,
            key=lambda item: (-item["error_count"], item["sample_id"]),
        )[: self.topk_samples]

        return {
            "num_samples": int(self.num_samples),
            "num_tokens": int(self.num_tokens),
            "token_accuracy": float(token_acc),
            "full_sequence_exact_match": float(exact),
            "num_exact": int(self.num_exact),
            "num_non_exact": int(non_exact),
            "one_error_only": int(one_error_only),
            "one_error_share_among_non_exact": float(one_error_share_among_nonexact),
            "sample_error_histogram": {
                str(k): int(v) for k, v in sorted(self.sample_error_hist.items(), key=lambda x: x[0])
            },
            "family_stats": [
                {"family": name, **stats} for name, stats in sorted_family_stats
            ],
            "per_position": [
                {"position": int(pos), **stats} for pos, stats in sorted_pos_stats
            ],
            "per_variable": [
                {"var_name": name, **stats} for name, stats in sorted_var_stats
            ],
            "top_cost_sensitive_variables": top_cost_sensitive[:50],
            "hardest_samples": hardest_samples,
        }


@torch.no_grad()
def _run_model(
    model: LatentParamVAE,
    batch: Dict[str, Any],
    decoder_input_ids: torch.Tensor,
    decoder_var_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    out = model(
        batch["encoder_token_ids"],
        batch["encoder_var_ids"],
        decoder_input_ids,
        decoder_var_ids,
        pad_token_id=pad_id,
    )
    return out.logits


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
        raise RuntimeError(f"No legal candidates available for {var_name}")
    return int(candidate_values[0])


def _fixed_token_id_for_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    value: int,
) -> int:
    token = tokenizer.value_to_token(var_name, int(value))
    return int(tokenizer.token_to_id.get(token, tokenizer.unk_id))


def _build_teacher_forcing_candidate_masks_from_records(
    batch: Dict[str, Any],
    records: Sequence[JsonSampleRecord],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    debug_invalid_step: bool = True,
) -> torch.Tensor:
    masks = batch.get("candidate_masks")
    if masks is not None:
        return masks

    target_ids: torch.Tensor = batch["target_ids"]
    ordered_names: List[List[str]] = batch["ordered_param_names"]
    ordered_values: List[List[int]] = batch["ordered_param_values"]
    sample_ids: List[str] = batch["sample_ids"]

    bsz, max_len = target_ids.shape
    vocab_size = len(tokenizer.id_to_token)
    masks = torch.zeros((bsz, max_len, vocab_size), dtype=torch.bool, device=device)

    for i, record in enumerate(records):
        oracle = registry.build_oracle_from_record(record)
        names = ordered_names[i]
        values = ordered_values[i]

        for t, (name, value) in enumerate(zip(names, values)):
            try:
                candidates = oracle.candidate_values(name)
                masks[i, t] = tokenizer.candidate_mask_from_values(name, candidates, device=device)

                gold_token = tokenizer.value_to_token(name, value)
                gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
                if debug_invalid_step and not masks[i, t, gold_id]:
                    print(
                        f"[invalid-step] sample={sample_ids[i]} step={t} var={name} "
                        f"gold={value} candidates={candidates}"
                    )
                    raise ValueError("gold value is outside oracle candidates")

                oracle.assign(name, value)
            except Exception:  # pylint: disable=broad-except
                gold_token = tokenizer.value_to_token(name, value)
                gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
                masks[i, t] = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                masks[i, t, gold_id] = True
                for rem_t, rem_name, rem_value in zip(
                    range(t + 1, len(names)),
                    names[t + 1:],
                    values[t + 1:],
                ):
                    rem_gold_token = tokenizer.value_to_token(rem_name, rem_value)
                    rem_gold_id = tokenizer.token_to_id.get(rem_gold_token, tokenizer.unk_id)
                    masks[i, rem_t] = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                    masks[i, rem_t, rem_gold_id] = True
                break

        for t in range(len(names), max_len):
            masks[i, t] = tokenizer.pad_only_mask(device=device)

    return masks


@torch.no_grad()
def _greedy_decode_batch_from_records(
    model,
    samples,
    records: Sequence[JsonSampleRecord],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
):
    if not samples:
        return []

    model.eval()
    batch = collate_prepared_samples(samples, tokenizer)
    enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
    enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, z, memory = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)

    oracles = [registry.build_oracle_from_record(record) for record in records]
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

        masked_logits = step_logits.masked_fill(~step_masks, float("-inf"))
        pred_token_ids = torch.argmax(masked_logits, dim=-1).tolist()

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

    results = []
    for sample_idx, sample in enumerate(samples):
        predicted_dict = {
            name: int(value)
            for name, value in zip(sample.ordered_param_names, decoded_values[sample_idx])
        }
        results.append(
            {
                "predicted_param_dict": predicted_dict,
                "predicted_token_ids": list(decoded_token_ids[sample_idx]),
            }
        )
    return results


def _add_sample_stats(
    analyzer: SplitAnalyzer,
    *,
    sample_id: str,
    ordered_names: Sequence[str],
    gold_values: Sequence[int],
    pred_values: Sequence[int],
    cost: float | None,
    position_ids: Sequence[int] | None = None,
) -> None:
    correctness = [int(p) == int(g) for p, g in zip(pred_values, gold_values)]
    if position_ids is None:
        position_ids = list(range(len(correctness)))
    wrong_indices = [idx for idx, ok in enumerate(correctness) if not ok]
    wrong_positions = [int(position_ids[idx]) for idx in wrong_indices]
    wrong_vars = [ordered_names[idx] for idx in wrong_indices]
    analyzer.add_sample(
        sample_id=sample_id,
        wrong_vars=wrong_vars,
        wrong_positions=wrong_positions,
        ordered_var_names=ordered_names,
        correctness=correctness,
        cost=cost,
        position_ids=position_ids,
    )


@torch.no_grad()
def analyze_reconstruction(
    model: LatentParamVAE,
    dataset,
    records: Sequence[JsonSampleRecord],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
    topk_samples: int,
) -> Dict[str, Any]:
    model.eval()
    full_tf_analyzer = SplitAnalyzer(topk_samples=topk_samples)
    compressed_tf_analyzer = SplitAnalyzer(topk_samples=topk_samples)
    full_ar_analyzer = SplitAnalyzer(topk_samples=topk_samples)
    compressed_ar_analyzer = SplitAnalyzer(topk_samples=topk_samples)

    samples = list(dataset.samples)
    if len(samples) != len(records):
        raise ValueError(
            "dataset samples and records must have the same length for record-aligned analysis: "
            f"samples={len(samples)} records={len(records)}"
        )
    stride = max(int(batch_size), 1)

    for start in range(0, len(samples), stride):
        batch_samples = samples[start:start + stride]
        batch_records = records[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        batch = _batch_to_device(batch, device)

        candidate_masks = _build_teacher_forcing_candidate_masks_from_records(
            batch,
            batch_records,
            registry,
            tokenizer,
            device=device,
            debug_invalid_step=False,
        )
        singleton_mask = _build_singleton_position_mask(
            batch["target_ids"],
            candidate_masks,
            tokenizer.pad_id,
        )
        keep_mask = batch["target_ids"].ne(tokenizer.pad_id) & (~singleton_mask)
        compressed = _compress_teacher_forcing_batch(batch, candidate_masks, tokenizer)

        logits_full = _run_model(
            model,
            batch,
            batch["decoder_input_ids"],
            batch["decoder_var_ids"],
            tokenizer.pad_id,
        )
        pred_full = torch.argmax(
            logits_full.masked_fill(~candidate_masks, float("-inf")),
            dim=-1,
        )

        logits_compressed = _run_model(
            model,
            batch,
            compressed["decoder_input_ids"],
            compressed["decoder_var_ids"],
            tokenizer.pad_id,
        )
        pred_compressed = torch.argmax(
            logits_compressed.masked_fill(~compressed["candidate_masks"], float("-inf")),
            dim=-1,
        )

        ar_results = _greedy_decode_batch_from_records(
            model,
            batch_samples,
            batch_records,
            registry,
            tokenizer,
            device,
        )

        for row_idx, (sample, ar_result) in enumerate(zip(batch_samples, ar_results)):
            ordered_names = list(sample.ordered_param_names)
            gold_values = [int(v) for v in sample.ordered_param_values]
            cost = None if sample.cost is None else float(sample.cost)

            # full teacher forcing
            seq_len = len(ordered_names)
            gold_full = batch["target_ids"][row_idx, :seq_len]
            pred_row_full = pred_full[row_idx, :seq_len]
            correctness_full = [bool(x) for x in pred_row_full.eq(gold_full).tolist()]
            wrong_positions_full = [idx for idx, ok in enumerate(correctness_full) if not ok]
            wrong_vars_full = [ordered_names[idx] for idx in wrong_positions_full]
            full_tf_analyzer.add_sample(
                sample_id=sample.sample_id,
                wrong_vars=wrong_vars_full,
                wrong_positions=wrong_positions_full,
                ordered_var_names=ordered_names,
                correctness=correctness_full,
                cost=cost,
            )

            # compressed teacher forcing (matches current validation TF metric path)
            kept_positions = torch.nonzero(keep_mask[row_idx], as_tuple=False).flatten().tolist()
            comp_len = len(kept_positions)
            comp_var_names = [ordered_names[pos] for pos in kept_positions]
            gold_comp = compressed["target_ids"][row_idx, :comp_len]
            pred_row_comp = pred_compressed[row_idx, :comp_len]
            correctness_comp = [bool(x) for x in pred_row_comp.eq(gold_comp).tolist()]
            wrong_positions_comp = [idx for idx, ok in enumerate(correctness_comp) if not ok]
            wrong_vars_comp = [comp_var_names[idx] for idx in wrong_positions_comp]
            compressed_tf_analyzer.add_sample(
                sample_id=sample.sample_id,
                wrong_vars=wrong_vars_comp,
                wrong_positions=[kept_positions[idx] for idx in wrong_positions_comp],
                ordered_var_names=comp_var_names,
                correctness=correctness_comp,
                cost=cost,
                position_ids=kept_positions,
            )

            # autoregressive full / compressed
            pred_values_full = [int(ar_result["predicted_param_dict"][name]) for name in ordered_names]
            _add_sample_stats(
                full_ar_analyzer,
                sample_id=sample.sample_id,
                ordered_names=ordered_names,
                gold_values=gold_values,
                pred_values=pred_values_full,
                cost=cost,
            )

            comp_gold_values = [gold_values[pos] for pos in kept_positions]
            comp_pred_values = [pred_values_full[pos] for pos in kept_positions]
            _add_sample_stats(
                compressed_ar_analyzer,
                sample_id=sample.sample_id,
                ordered_names=comp_var_names,
                gold_values=comp_gold_values,
                pred_values=comp_pred_values,
                cost=cost,
                position_ids=kept_positions,
            )

    return {
        "full_teacher_forcing": full_tf_analyzer.to_dict(),
        "compressed_teacher_forcing": compressed_tf_analyzer.to_dict(),
        "full_autoregressive": full_ar_analyzer.to_dict(),
        "compressed_autoregressive": compressed_ar_analyzer.to_dict(),
    }


def _load_model_and_bundle(
    checkpoint_path: str | Path,
    *,
    device: str,
    json_paths: Sequence[str] | None,
    network_info_folder: str | None,
):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")

    cfg = _dict_to_namespace(payload["config"])
    if json_paths:
        cfg.data.json_paths = list(json_paths)
    if network_info_folder is not None:
        cfg.data.network_info_folder = str(network_info_folder)

    if hasattr(cfg.train, "precompute_candidate_masks"):
        cfg.train.precompute_candidate_masks = False

    registry = GeneratorRegistry(cfg.data.network_info_folder)
    bundle = build_dataset_bundle(cfg, registry)

    tokenizer = ParamTokenizer.from_checkpoint_payload(payload)
    _assert_same_tokenizer(tokenizer, bundle.tokenizer)

    model_cfg = _make_model_cfg_namespace(payload["config"]["model"])
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=model_cfg,
    )
    model.load_state_dict(payload["model_state"])
    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    model.eval()

    return payload, cfg, registry, bundle, tokenizer, model, torch_device


def _select_dataset_and_records(bundle, split: str):
    split = str(split).lower()
    if split == "train":
        return bundle.train_dataset, list(bundle.train_records)
    if split == "val":
        return bundle.val_dataset, list(bundle.val_records)
    if split == "test":
        return bundle.test_dataset, list(bundle.test_records)
    raise ValueError(f"Unknown split: {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--json-path", type=str, action="append", default=None)
    parser.add_argument("--network-info-folder", type=str, default=None)
    parser.add_argument("--topk-samples", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, cfg, registry, bundle, tokenizer, model, device = _load_model_and_bundle(
        args.checkpoint,
        device=args.device,
        json_paths=args.json_path,
        network_info_folder=args.network_info_folder,
    )
    dataset, records = _select_dataset_and_records(bundle, args.split)
    report = analyze_reconstruction(
        model=model,
        dataset=dataset,
        records=records,
        registry=registry,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size or int(getattr(cfg.eval, "batch_size", 128)),
        topk_samples=args.topk_samples,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": str(args.split),
        "num_split_samples": int(len(dataset.samples)),
        "config_data_json_paths": list(cfg.data.json_paths),
        "report": report,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[saved] {output_path}")
    print(text)


if __name__ == "__main__":
    main()
