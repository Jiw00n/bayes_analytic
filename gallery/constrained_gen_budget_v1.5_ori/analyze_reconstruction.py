from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch

try:
    from latent_model_budget.adapter import GeneratorRegistry
    from latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from latent_model_budget.model import LatentParamVAE
    from latent_model_budget.tokenizer import ParamTokenizer
    from latent_model_budget.train_eval import (
        _batch_to_device,
        _build_singleton_position_mask,
        _build_teacher_forcing_candidate_masks,
        _compress_teacher_forcing_batch,
    )
except ImportError:  # pragma: no cover
    import sys

    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    from .latent_model_budget.adapter import GeneratorRegistry
    from .latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from .latent_model_budget.model import LatentParamVAE
    from .latent_model_budget.tokenizer import ParamTokenizer
    from .latent_model_budget.train_eval import (
        _batch_to_device,
        _build_singleton_position_mask,
        _build_teacher_forcing_candidate_masks,
        _compress_teacher_forcing_batch,
    )


def _dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    return obj


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
    ) -> None:
        self.num_samples += 1
        error_count = int(sum(1 for ok in correctness if not ok))
        self.sample_error_hist[error_count] += 1
        if error_count == 0:
            self.num_exact += 1
        self.num_tokens += len(correctness)
        self.num_correct_tokens += int(sum(1 for ok in correctness if ok))

        for pos, (var_name, ok) in enumerate(zip(ordered_var_names, correctness)):
            self.var_stats[var_name].update(correct=bool(ok), cost=cost)
            self.pos_stats[int(pos)].update(correct=bool(ok), cost=cost)
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


@torch.no_grad()
def analyze_reconstruction(
    model: LatentParamVAE,
    dataset,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
    topk_samples: int,
) -> Dict[str, Any]:
    model.eval()
    full_analyzer = SplitAnalyzer(topk_samples=topk_samples)
    compressed_analyzer = SplitAnalyzer(topk_samples=topk_samples)

    samples = list(dataset.samples)
    stride = max(int(batch_size), 1)

    for start in range(0, len(samples), stride):
        batch_samples = samples[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        batch = _batch_to_device(batch, device)

        candidate_masks = _build_teacher_forcing_candidate_masks(
            batch,
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

        for row_idx, sample in enumerate(batch_samples):
            ordered_names = list(sample.ordered_param_names)
            cost = None
            if sample.cost is not None:
                cost = float(sample.cost)

            # full-sequence stats
            seq_len = len(ordered_names)
            gold_full = batch["target_ids"][row_idx, :seq_len]
            pred_row_full = pred_full[row_idx, :seq_len]
            correctness_full = [bool(x) for x in pred_row_full.eq(gold_full).tolist()]
            wrong_positions_full = [idx for idx, ok in enumerate(correctness_full) if not ok]
            wrong_vars_full = [ordered_names[idx] for idx in wrong_positions_full]
            full_analyzer.add_sample(
                sample_id=sample.sample_id,
                wrong_vars=wrong_vars_full,
                wrong_positions=wrong_positions_full,
                ordered_var_names=ordered_names,
                correctness=correctness_full,
                cost=cost,
            )

            # compressed stats (this matches current validation TF metric path)
            kept_positions = torch.nonzero(keep_mask[row_idx], as_tuple=False).flatten().tolist()
            comp_len = len(kept_positions)
            comp_var_names = [ordered_names[pos] for pos in kept_positions]
            gold_comp = compressed["target_ids"][row_idx, :comp_len]
            pred_row_comp = pred_compressed[row_idx, :comp_len]
            correctness_comp = [bool(x) for x in pred_row_comp.eq(gold_comp).tolist()]
            wrong_positions_comp = [idx for idx, ok in enumerate(correctness_comp) if not ok]
            wrong_vars_comp = [comp_var_names[idx] for idx in wrong_positions_comp]
            compressed_analyzer.add_sample(
                sample_id=sample.sample_id,
                wrong_vars=wrong_vars_comp,
                wrong_positions=wrong_positions_comp,
                ordered_var_names=comp_var_names,
                correctness=correctness_comp,
                cost=cost,
            )

    return {
        "full_teacher_forcing": full_analyzer.to_dict(),
        "compressed_teacher_forcing": compressed_analyzer.to_dict(),
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

    # Analysis does not need dataset-time precompute.
    if hasattr(cfg.train, "precompute_candidate_masks"):
        cfg.train.precompute_candidate_masks = False

    registry = GeneratorRegistry(cfg.data.network_info_folder)
    bundle = build_dataset_bundle(cfg, registry)

    tokenizer = ParamTokenizer.from_state_dict(payload["tokenizer"])
    _assert_same_tokenizer(tokenizer, bundle.tokenizer)

    model_cfg = _dict_to_namespace(payload["config"]["model"])
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


def _select_dataset(bundle, split: str):
    split = str(split).lower()
    if split == "train":
        return bundle.train_dataset
    if split == "val":
        return bundle.val_dataset
    if split == "test":
        return bundle.test_dataset
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
    dataset = _select_dataset(bundle, args.split)
    report = analyze_reconstruction(
        model=model,
        dataset=dataset,
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
