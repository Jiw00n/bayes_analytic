from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm.auto import tqdm

try:
    from latent_model_budget.adapter import GeneratorRegistry
    from latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from latent_model_budget.inference import greedy_decode_batch
    from latent_model_budget.model import LatentParamVAE
    from latent_model_budget.tokenizer import ParamTokenizer
    from latent_model_budget.train_eval import (
        _batch_to_device,
        _build_teacher_forcing_candidate_masks,
        _teacher_forcing_accuracy_stats,
    )
except ImportError:  # pragma: no cover
    import sys

    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    from latent_model_budget.adapter import GeneratorRegistry
    from latent_model_budget.dataset import build_dataset_bundle, collate_prepared_samples
    from latent_model_budget.inference import greedy_decode_batch
    from latent_model_budget.model import LatentParamVAE
    from latent_model_budget.tokenizer import ParamTokenizer
    from latent_model_budget.train_eval import (
        _batch_to_device,
        _build_teacher_forcing_candidate_masks,
        _teacher_forcing_accuracy_stats,
    )


class _AttrDict(dict):
    """dict subclass that also supports attribute-style access.

    Downstream code uses both ``cfg.data.hw_param`` (attribute) and
    ``dict(cfg.data.hw_param)`` (dict-iteration) on the same node, so we
    cannot use SimpleNamespace.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({k: _dict_to_namespace(v) for k, v in obj.items()})
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


def _build_token_remap(
    built: ParamTokenizer, saved: ParamTokenizer
) -> tuple[Dict[int, int], Dict[int, int]]:
    """Return (token_id_remap, var_id_remap) mapping built-tokenizer integer
    IDs to saved-tokenizer integer IDs by matching the string representation.

    A built ID with no string-equivalent in ``saved`` is omitted from the
    mapping; the caller must skip any sample whose IDs touch such entries.
    """
    saved_token_to_id = {tok: idx for idx, tok in enumerate(saved.id_to_token)}
    saved_var_to_id = {var: idx for idx, var in enumerate(saved.id_to_var)}
    token_remap: Dict[int, int] = {}
    for built_id, tok in enumerate(built.id_to_token):
        sid = saved_token_to_id.get(tok)
        if sid is not None:
            token_remap[int(built_id)] = int(sid)
    var_remap: Dict[int, int] = {}
    for built_id, var in enumerate(built.id_to_var):
        sid = saved_var_to_id.get(var)
        if sid is not None:
            var_remap[int(built_id)] = int(sid)
    return token_remap, var_remap


def _remap_candidate_masks(
    masks: torch.Tensor,
    token_remap: Dict[int, int],
    saved_vocab_size: int,
    saved_unk_id: int,
) -> torch.Tensor:
    """Remap a [seq_len, built_vocab] bool mask onto saved_vocab indexing.

    Each ``built_id`` that has a saved equivalent moves its column. Positions
    whose entire built-vocab mask had at least one True but no remapped True
    survive only via ``saved_unk_id`` (mirrors ``candidate_mask_from_values``
    UNK fallback behaviour).
    """
    seq_len = masks.shape[0]
    new_masks = torch.zeros((seq_len, saved_vocab_size), dtype=torch.bool)
    for built_id, saved_id in token_remap.items():
        new_masks[:, saved_id] = masks[:, built_id]
    had_any = masks.any(dim=-1)
    has_any = new_masks.any(dim=-1)
    fallback = had_any & ~has_any
    if fallback.any():
        new_masks[fallback, saved_unk_id] = True
    return new_masks


def _remap_sample(
    sample,
    token_remap: Dict[int, int],
    var_remap: Dict[int, int],
    *,
    token_unk_id: int,
    var_pad_id: int,
    saved_vocab_size: Optional[int] = None,
):
    """Return a new PreparedSample with all integer token/var IDs remapped to
    the saved tokenizer's IDs. Unmapped token IDs become ``token_unk_id``;
    unmapped var IDs become ``var_pad_id``. Returns the unmapped-touch count
    so callers can track partial drift per sample.
    """
    from copy import copy as _shallow

    touched = {"tok": 0, "var": 0}

    def _map_tokens(seq):
        out = []
        for v in seq:
            mapped = token_remap.get(int(v))
            if mapped is None:
                touched["tok"] += 1
                out.append(int(token_unk_id))
            else:
                out.append(int(mapped))
        return out

    def _map_vars(seq):
        out = []
        for v in seq:
            mapped = var_remap.get(int(v))
            if mapped is None:
                touched["var"] += 1
                out.append(int(var_pad_id))
            else:
                out.append(int(mapped))
        return out

    out = _shallow(sample)
    out.encoder_token_ids = _map_tokens(sample.encoder_token_ids)
    out.decoder_input_ids = _map_tokens(sample.decoder_input_ids)
    out.target_ids = _map_tokens(sample.target_ids)
    out.shape_token_ids = _map_tokens(sample.shape_token_ids)
    out.extent_token_ids = _map_tokens(sample.extent_token_ids)
    out.encoder_var_ids = _map_vars(sample.encoder_var_ids)
    out.decoder_var_ids = _map_vars(sample.decoder_var_ids)
    out.shape_var_ids = _map_vars(sample.shape_var_ids)
    out.extent_var_ids = _map_vars(sample.extent_var_ids)
    if (
        sample.candidate_masks is not None
        and saved_vocab_size is not None
    ):
        out.candidate_masks = _remap_candidate_masks(
            sample.candidate_masks,
            token_remap,
            saved_vocab_size,
            int(token_unk_id),
        )
    else:
        out.candidate_masks = None
    return out, touched["tok"], touched["var"]


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
        self.pos_var_names: dict[int, Counter[str]] = defaultdict(Counter)
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
            self.pos_var_names[int(pos)][str(var_name)] += 1
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
                {
                    "position": int(pos),
                    "var_name": (
                        next(iter(self.pos_var_names[int(pos)]))
                        if len(self.pos_var_names[int(pos)]) == 1
                        else None
                    ),
                    **stats,
                }
                for pos, stats in sorted_pos_stats
            ],
            "per_variable": [
                {"var_name": name, **stats} for name, stats in sorted_var_stats
            ],
            "top_cost_sensitive_variables": top_cost_sensitive[:50],
            "hardest_samples": hardest_samples,
        }


def _add_sample_stats(
    analyzer: SplitAnalyzer,
    *,
    sample_id: str,
    ordered_names: Sequence[str],
    gold_values: Sequence[int],
    pred_values: Sequence[int],
    cost: float | None,
) -> None:
    correctness = [int(p) == int(g) for p, g in zip(pred_values, gold_values)]
    wrong_positions = [idx for idx, ok in enumerate(correctness) if not ok]
    wrong_vars = [ordered_names[idx] for idx in wrong_positions]
    analyzer.add_sample(
        sample_id=sample_id,
        wrong_vars=wrong_vars,
        wrong_positions=wrong_positions,
        ordered_var_names=ordered_names,
        correctness=correctness,
        cost=cost,
    )


TaskKey = tuple  # (task_index_or_None, workload_key_or_None, target_kind_or_None)


def _task_key(sample) -> TaskKey:
    return (
        None if sample.task_index is None else int(sample.task_index),
        None if sample.workload_key is None else str(sample.workload_key),
        None if sample.target_kind is None else str(sample.target_kind),
    )


def _task_key_sort(key: TaskKey):
    ti, wk, tk = key
    return (ti is None, ti if ti is not None else -1, wk or "", tk or "")


def _filter_samples(
    samples: Sequence[Any],
    task_indices: Optional[Sequence[int]],
    workload_substrs: Optional[Sequence[str]],
) -> List[Any]:
    out = list(samples)
    if task_indices:
        wanted = {int(t) for t in task_indices}
        out = [s for s in out if s.task_index is not None and int(s.task_index) in wanted]
    if workload_substrs:
        needles = [str(x) for x in workload_substrs]
        out = [
            s
            for s in out
            if s.workload_key is not None
            and any(n in str(s.workload_key) for n in needles)
        ]
    return out


@torch.no_grad()
def _tf_metrics_for_batch(
    model: LatentParamVAE,
    batch_samples: Sequence[Any],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Run a single teacher-forcing pass over ``batch_samples`` and return
    (token_correct, token_total, exact_count, sample_total) as int counts.

    Mirrors the encoded-z TF eval used by ``train_eval.evaluate_teacher_forcing``
    so the resulting metric is directly comparable to ``val_full_sequence_exact_match``.
    Encoder uses ``deterministic=True`` (z = mu).
    """
    if not batch_samples:
        return 0, 0, 0, 0
    batch = collate_prepared_samples(batch_samples, tokenizer)
    batch = _batch_to_device(batch, device)
    enc_ids = batch["encoder_token_ids"]
    enc_var_ids = batch["encoder_var_ids"]
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, z, memory = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
    candidate_masks = _build_teacher_forcing_candidate_masks(
        batch, registry, tokenizer, device=device, debug_invalid_step=False,
    )
    decoder_input_ids = batch["decoder_input_ids"]
    decoder_var_ids = batch["decoder_var_ids"]
    target_ids = batch["target_ids"]
    dec_pad = decoder_input_ids.eq(tokenizer.pad_id)
    logits = model.decode(decoder_input_ids, decoder_var_ids, memory, z, dec_pad)
    tc, tt, ec, st = _teacher_forcing_accuracy_stats(
        logits, target_ids, candidate_masks, tokenizer.pad_id,
    )
    return int(tc.item()), int(tt.item()), int(ec.item()), int(st.item())


@torch.no_grad()
def analyze_reconstruction_ar(
    model: LatentParamVAE,
    samples: Sequence[Any],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
    topk_samples: int,
    *,
    output_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Free-running (autoregressive) reconstruction analysis.

    Returns a report with an overall analyzer plus per-task analyzers keyed by
    ``task_index``. Each task entry also carries (workload_key, target_kind)
    for readability.
    """
    model.eval()
    overall = SplitAnalyzer(topk_samples=topk_samples)
    per_task: Dict[TaskKey, SplitAnalyzer] = {}
    # Teacher-forcing counters per task: (token_correct, token_total, exact, sample_total).
    tf_per_task: Dict[tuple, List[int]] = {}
    tf_overall = [0, 0, 0, 0]

    stride = max(int(batch_size), 1)
    samples = list(samples)
    total = len(samples)

    # Group by (task_index, workload_key, target_kind, sketch_index): each
    # batch must contain samples from a single task only, so the oracle
    # generator and its candidate/mask caches stay hot across the per-step
    # Python loop. Mixed-task batches waste the caches and leave GPU idle.
    task_groups: Dict[tuple, List[Any]] = {}
    task_order: List[tuple] = []
    for s in samples:
        key = (
            s.task_index if s.task_index is not None else -1,
            str(s.workload_key) if s.workload_key is not None else "",
            str(s.target_kind) if s.target_kind is not None else "",
            int(s.sketch_index),
        )
        if key not in task_groups:
            task_groups[key] = []
            task_order.append(key)
        task_groups[key].append(s)

    # Pre-warm sketch cache: registry.build_oracle hits TVM SketchPolicy on
    # first use per (workload_key, target_kind, sketch_index). Without this,
    # the first batch stalls (looks like GPU underutilization) while sketches
    # are built lazily inside greedy_decode_batch.
    if task_order:
        warm = tqdm(
            task_order,
            desc=f"sketch warm-up ({len(task_order)} task/sketch combos)",
            unit="combo",
        )
        for key in warm:
            s = task_groups[key][0]
            registry.build_oracle(
                task_index=s.task_index,
                sketch_index=s.sketch_index,
                workload_key=s.workload_key,
                target_kind=s.target_kind,
            )

    num_batches = sum(
        (len(grp) + stride - 1) // stride for grp in task_groups.values()
    )
    pbar = tqdm(
        total=num_batches,
        desc=f"Free-running",
        unit="batch",
    )
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    samples_seen = 0
    for key in task_order:
        grp = task_groups[key]
        if key not in tf_per_task:
            tf_per_task[key] = [0, 0, 0, 0]
        for start in range(0, len(grp), stride):
            batch_samples = grp[start:start + stride]
            ar_results = greedy_decode_batch(
                model, batch_samples, registry, tokenizer, device
            )
            tc, tt, ec, st = _tf_metrics_for_batch(
                model, batch_samples, registry, tokenizer, device
            )
            tf_per_task[key][0] += tc
            tf_per_task[key][1] += tt
            tf_per_task[key][2] += ec
            tf_per_task[key][3] += st
            tf_overall[0] += tc
            tf_overall[1] += tt
            tf_overall[2] += ec
            tf_overall[3] += st
            samples_seen += len(batch_samples)
            pbar.update(1)
            pbar.set_postfix(
                task=str(key[0]),
                bsz=len(batch_samples),
                samples=f"{samples_seen}/{total}",
                exact=f"{overall.num_exact}/{overall.num_samples}" if overall.num_samples else "0/0",
                tf=f"{tf_overall[2]}/{tf_overall[3]}" if tf_overall[3] else "0/0",
            )

            for sample, ar_result in zip(batch_samples, ar_results):
                ordered_names = list(sample.ordered_param_names)
                gold_values = [int(v) for v in sample.ordered_param_values]
                cost = None if sample.cost is None else float(sample.cost)
                pred_values = [
                    int(ar_result.predicted_param_dict[name]) for name in ordered_names
                ]

                _add_sample_stats(
                    overall,
                    sample_id=sample.sample_id,
                    ordered_names=ordered_names,
                    gold_values=gold_values,
                    pred_values=pred_values,
                    cost=cost,
                )

                tkey = _task_key(sample)
                if tkey not in per_task:
                    per_task[tkey] = SplitAnalyzer(topk_samples=topk_samples)
                _add_sample_stats(
                    per_task[tkey],
                    sample_id=sample.sample_id,
                    ordered_names=ordered_names,
                    gold_values=gold_values,
                    pred_values=pred_values,
                    cost=cost,
                )

        # Task-level dump on completion of this task group.
        if output_dir is not None:
            ti = key[0] if key[0] != -1 else None
            wk = key[1] or None
            tk = key[2] or None
            sample = grp[0]
            tkey = _task_key(sample)
            tc, tt, ec, st = tf_per_task[key]
            task_report = {
                "task_index": ti,
                "workload_key": wk,
                "target_kind": tk,
                "sketch_index": int(key[3]),
                "num_samples": len(grp),
                **(metadata or {}),
                "teacher_forcing": {
                    "token_accuracy": float(tc) / max(tt, 1),
                    "full_sequence_exact_match": float(ec) / max(st, 1),
                    "token_correct": int(tc),
                    "token_total": int(tt),
                    "exact_count": int(ec),
                    "sample_total": int(st),
                },
                "report": per_task[tkey].to_dict(),
            }
            fname_task = "none" if ti is None else str(ti)
            out_path = Path(output_dir) / f"task_{fname_task}_sk{int(key[3])}.json"
            out_path.write_text(
                json.dumps(task_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tqdm.write(f"[saved] {out_path}")
    pbar.close()

    # Build a quick lookup from (task_index, workload_key, target_kind) → tf counters,
    # collapsing the optional sketch_index from the grouping key.
    tf_by_tkey: Dict[TaskKey, List[int]] = {}
    for grp_key, counters in tf_per_task.items():
        ti = grp_key[0] if grp_key[0] != -1 else None
        wk = grp_key[1] or None
        tk = grp_key[2] or None
        agg = tf_by_tkey.setdefault((ti, wk, tk), [0, 0, 0, 0])
        for i in range(4):
            agg[i] += counters[i]

    per_task_report = []
    for tkey in sorted(per_task.keys(), key=_task_key_sort):
        ti, wk, tk = tkey
        tc, tt, ec, st = tf_by_tkey.get(tkey, [0, 0, 0, 0])
        entry = {
            "task_index": ti,
            "workload_key": wk,
            "target_kind": tk,
            "teacher_forcing": {
                "token_accuracy": float(tc) / max(tt, 1),
                "full_sequence_exact_match": float(ec) / max(st, 1),
                "token_correct": int(tc),
                "token_total": int(tt),
                "exact_count": int(ec),
                "sample_total": int(st),
            },
            **per_task[tkey].to_dict(),
        }
        per_task_report.append(entry)

    overall_dict = overall.to_dict()
    overall_dict["teacher_forcing"] = {
        "token_accuracy": float(tf_overall[0]) / max(tf_overall[1], 1),
        "full_sequence_exact_match": float(tf_overall[2]) / max(tf_overall[3], 1),
        "token_correct": int(tf_overall[0]),
        "token_total": int(tf_overall[1]),
        "exact_count": int(tf_overall[2]),
        "sample_total": int(tf_overall[3]),
    }

    return {
        "overall": overall_dict,
        "per_task": per_task_report,
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

    gen_cfg = getattr(cfg, "generator", None)
    raw_hw_param = getattr(gen_cfg, "hw_param", None) if gen_cfg is not None else None
    hw_param_dict = dict(raw_hw_param) if raw_hw_param else None
    raw_disable = getattr(gen_cfg, "disable_constraint", None) if gen_cfg is not None else None
    disable_constraint_list = list(raw_disable) if raw_disable else None
    print(
        f"[registry] hw_param={hw_param_dict} "
        f"disable_constraint={disable_constraint_list}"
    )
    registry = GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=hw_param_dict,
        disable_constraint=disable_constraint_list,
    )
    bundle = build_dataset_bundle(cfg, registry)

    saved_tokenizer = ParamTokenizer.from_state_dict(payload["tokenizer"])
    if (
        saved_tokenizer.id_to_token != bundle.tokenizer.id_to_token
        or saved_tokenizer.id_to_var != bundle.tokenizer.id_to_var
    ):
        token_remap, var_remap = _build_token_remap(bundle.tokenizer, saved_tokenizer)
        unmapped_tokens = sorted(
            set(bundle.tokenizer.id_to_token) - set(saved_tokenizer.id_to_token)
        )
        unmapped_vars = sorted(
            set(bundle.tokenizer.id_to_var) - set(saved_tokenizer.id_to_var)
        )
        print(
            f"[tokenizer] vocab drift: {len(unmapped_tokens)} unmapped tokens, "
            f"{len(unmapped_vars)} unmapped vars; samples touching these will be skipped"
        )
        for name, ds in (
            ("train", bundle.train_dataset),
            ("val", bundle.val_dataset),
            ("test", bundle.test_dataset),
        ):
            kept = []
            tok_touch_per_task: Dict[Optional[int], int] = {}
            samples_with_touch_per_task: Dict[Optional[int], int] = {}
            count_per_task: Dict[Optional[int], int] = {}
            for sample in ds.samples:
                ti = None if sample.task_index is None else int(sample.task_index)
                remapped, tok_hits, var_hits = _remap_sample(
                    sample,
                    token_remap,
                    var_remap,
                    token_unk_id=int(saved_tokenizer.unk_id),
                    var_pad_id=int(saved_tokenizer.var_pad_id),
                    saved_vocab_size=len(saved_tokenizer.id_to_token),
                )
                kept.append(remapped)
                count_per_task[ti] = count_per_task.get(ti, 0) + 1
                if tok_hits or var_hits:
                    samples_with_touch_per_task[ti] = (
                        samples_with_touch_per_task.get(ti, 0) + 1
                    )
                    tok_touch_per_task[ti] = tok_touch_per_task.get(ti, 0) + tok_hits
            ds.samples = kept
            total_touched_samples = sum(samples_with_touch_per_task.values())
            print(
                f"[tokenizer] {name}: kept {len(kept)} samples; "
                f"{total_touched_samples} sample(s) had at least one position remapped to UNK"
            )
            for ti in sorted(count_per_task, key=lambda x: (x is None, x or -1)):
                touched = samples_with_touch_per_task.get(ti, 0)
                tok_hits = tok_touch_per_task.get(ti, 0)
                if touched == 0:
                    continue
                total = count_per_task[ti]
                print(
                    f"  [tokenizer]   task={ti}: total={total} "
                    f"samples_with_unk={touched} "
                    f"avg_unk_positions={tok_hits/total:.2f}"
                )
        bundle.tokenizer = saved_tokenizer
    tokenizer = saved_tokenizer

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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--json-path", type=str, action="append", default=None)
    parser.add_argument("--network-info-folder", type=str, default=None)
    parser.add_argument("--topk-samples", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to dump per-task JSON files (one file per task as it "
            "finishes decoding)."
        ),
    )
    parser.add_argument(
        "--task",
        type=int,
        action="append",
        default=None,
        help=(
            "task_index to include (repeatable). If omitted, all tasks in the "
            "split are evaluated."
        ),
    )
    parser.add_argument(
        "--workload-key",
        type=str,
        action="append",
        default=None,
        help=(
            "Substring of workload_key to include (repeatable, OR semantics). "
            "Useful when JSON metadata has task_index=null."
        ),
    )
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

    samples = _filter_samples(dataset.samples, args.task, args.workload_key)
    if not samples:
        available_idx = sorted(
            {int(s.task_index) for s in dataset.samples if s.task_index is not None}
        )
        available_wk = sorted(
            {str(s.workload_key) for s in dataset.samples if s.workload_key is not None}
        )
        raise RuntimeError(
            f"No samples after filtering split={args.split} task={args.task} "
            f"workload_key={args.workload_key}. "
            f"Available task_index: {available_idx}. "
            f"#distinct workload_key in split: {len(available_wk)}."
        )

    per_task_metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": str(args.split),
        "decoding": "free_running_autoregressive",
    }
    report_body = analyze_reconstruction_ar(
        model=model,
        samples=samples,
        registry=registry,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size or int(getattr(cfg.eval, "batch_size", 128)),
        topk_samples=args.topk_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        metadata=per_task_metadata,
    )

    selected_tasks = (
        sorted({int(t) for t in args.task}) if args.task else None
    )
    print(
        f"[done] split={args.split} "
        f"samples={len(samples)}/{len(dataset.samples)} "
        f"selected_tasks={selected_tasks} "
        f"selected_workload_substrs={list(args.workload_key) if args.workload_key else None} "
        f"output_dir={args.output_dir}"
    )
    ov = report_body["overall"]
    tf = ov.get("teacher_forcing", {})
    print(
        f"[overall] num_samples={ov['num_samples']} "
        f"AR token_acc={ov['token_accuracy']:.4f} "
        f"AR exact_match={ov['full_sequence_exact_match']:.4f} "
        f"TF token_acc={tf.get('token_accuracy', 0.0):.4f} "
        f"TF exact_match={tf.get('full_sequence_exact_match', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
