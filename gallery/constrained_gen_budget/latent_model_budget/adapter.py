from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class JsonSampleRecord:
    sample_id: str
    json_path: str
    sketch_index: int
    params: Dict[str, int]
    cost: Optional[float]
    raw: dict
    workload_key: Optional[str] = None
    target_kind: Optional[str] = None
    target_model: Optional[str] = None
    task_desc: Optional[str] = None
    task_index: Optional[int] = None
    record_index: Optional[int] = None
    param_signature: Optional[Tuple[str, ...]] = None

ENABLED_CONSTRAINTS = (
    'shared_memory',
    'max_threads',
    'max_vthread',
    'innermost_split',
    'split_structure',
    # 'max_threads_per_block',
    # 'max_vector_bytes',
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _runtime_enabled_constraints():
    if _env_flag("CGB_USE_DEFAULT_ENABLED_CONSTRAINTS", False):
        return None
    return ENABLED_CONSTRAINTS


# -----------------------------------------------------------------------------
# JSON parsing
# -----------------------------------------------------------------------------


def _maybe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_budget_param_name(name: str) -> bool:
    return name.startswith("thread_budget") or name.startswith("vthread_budget")


def _is_modeled_param_name(name: str) -> bool:
    return name.startswith("sp_") or name.startswith("ur_") or _is_budget_param_name(name)


def _normalize_generator_signature(names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(name for name in names if not _is_budget_param_name(name))


def _extract_param_dict(payload: dict) -> Dict[str, int]:
    sym_map = payload.get("sym_map", {})
    params: Dict[str, int] = {}
    for name, value in sym_map.items():
        if not _is_modeled_param_name(str(name)):
            continue
        if value is None:
            continue
        params[str(name)] = int(value)
    if not params:
        raise ValueError("No concrete modeled parameters found in payload['sym_map']")
    return params


def _extract_task_signature(payload: dict) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
    meta = payload.get("meta", {}) or {}
    task = payload.get("task", {}) or {}

    task_index = _maybe_int(meta.get("task_index"))
    if task_index is None:
        task_index = _maybe_int(task.get("task_index"))

    workload_key = task.get("workload_key")
    if workload_key is not None:
        workload_key = str(workload_key)

    target_kind = task.get("target_kind")
    if target_kind is not None:
        target_kind = str(target_kind)

    target_model = task.get("target_model")
    if target_model is not None:
        target_model = str(target_model)

    task_desc = task.get("desc")
    if task_desc is not None:
        task_desc = str(task_desc)

    return task_index, workload_key, target_kind, target_model, task_desc


def _build_sample_id(
    path: Path,
    workload_key: Optional[str],
    target_kind: Optional[str],
    sketch_index: int,
    task_index: Optional[int],
    record_index: Optional[int] = None,
) -> str:
    suffix = "" if record_index is None else f"::r{int(record_index)}"
    if workload_key and target_kind:
        return f"{target_kind}:{workload_key}#s{int(sketch_index)}::{path.stem}{suffix}"
    if task_index is not None:
        return f"task{int(task_index)}_sketch{int(sketch_index)}_{path.stem}{suffix}"
    return f"unknown_task_sketch{int(sketch_index)}_{path.stem}{suffix}"


def _extract_measure_record_params(state) -> Tuple[Dict[str, int], Tuple[str, ...]]:
    params: Dict[str, int] = {}
    param_signature: List[str] = []

    for step_idx, step in enumerate(state.transform_steps):
        step_type = step.type_key.split(".")[-1]
        if step_type == "SplitStep":
            for length_idx, length in enumerate(step.lengths):
                name = f"sp_{step_idx}_{length_idx}"
                params[name] = int(length)
                param_signature.append(name)
            continue

        if step_type != "PragmaStep":
            continue

        pragma_type = str(step.pragma_type)
        if not pragma_type.startswith("auto_unroll_max_step$"):
            continue

        name = f"ur_{step_idx}"
        params[name] = int(pragma_type.split("$")[-1])
        param_signature.append(name)

    if not params:
        raise ValueError("No concrete modeled parameters found in measure record")

    return params, tuple(param_signature)


def _extract_measure_record_params_from_payload(payload: dict) -> Tuple[Dict[str, int], Tuple[str, ...]]:
    steps = payload["i"][1][1]
    params: Dict[str, int] = {}
    param_signature: List[str] = []

    for step_idx, step in enumerate(steps):
        if not isinstance(step, list) or not step:
            continue
        step_code = str(step[0])
        if step_code == "SP":
            lengths = step[4] if len(step) > 4 else []
            for length_idx, length in enumerate(lengths):
                name = f"sp_{step_idx}_{length_idx}"
                params[name] = int(length)
                param_signature.append(name)
            continue

        if step_code != "PR" or len(step) <= 3:
            continue
        pragma_type = str(step[3])
        if not pragma_type.startswith("auto_unroll_max_step$"):
            continue
        name = f"ur_{step_idx}"
        params[name] = int(pragma_type.split("$")[-1])
        param_signature.append(name)

    if not params:
        raise ValueError("No concrete modeled parameters found in measure record payload")

    return params, tuple(param_signature)


def _record_split_extent_overrides(record: JsonSampleRecord) -> Optional[Dict[int, int]]:
    """Extract per-SplitStep frozen extent from record's raw JSON.

    Sketch-time extents (e.g. vectorize splits on cache_read after FuseStep)
    are frozen in the raw record's SP step entry. Returned as a read-only
    override dict to be threaded into per-call extent lookups, avoiding any
    mutation of the shared cached generator.
    """
    raw_steps = record.raw.get("i", [None, [None, None]])[1][1]
    if not isinstance(raw_steps, list):
        return None
    overrides: Dict[int, int] = {}
    for step_idx, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, list) or not raw_step or raw_step[0] != "SP":
            continue
        if len(raw_step) < 4:
            continue
        try:
            overrides[int(step_idx)] = int(raw_step[3])
        except (TypeError, ValueError):
            continue
    return overrides or None


def _augment_record_budget_params(record: JsonSampleRecord, generator) -> None:
    params = record.params
    budget_specs = list(getattr(generator, "_budget_specs", ()))
    for spec in budget_specs:
        budget_name = str(spec["name"])
        if budget_name in params:
            params[budget_name] = int(params[budget_name])
            continue
        factor_names = [str(name) for name in spec.get("factor_names", ())]
        if not factor_names or any(name not in params for name in factor_names):
            continue
        budget_value = 1
        for factor_name in factor_names:
            budget_value *= int(params[factor_name])
        params[budget_name] = int(budget_value)

    full_order = list(generator.get_full_var_order_entries()["param_order"])
    ordered_names = [name for name in full_order if name in params]
    seen = set(ordered_names)
    ordered_names.extend(name for name in params.keys() if name not in seen)
    record.param_signature = tuple(ordered_names)


def _extract_search_task_signature(task) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    workload_key = getattr(task, "workload_key", None)
    if workload_key is not None:
        workload_key = str(workload_key)

    target = getattr(task, "target", None)
    target_kind = getattr(getattr(target, "kind", None), "name", None)
    if target_kind is None and target is not None:
        target_kind = getattr(target, "kind", None)
    if target_kind is not None:
        target_kind = str(target_kind)

    target_model = getattr(target, "model", None)
    if target_model is not None:
        target_model = str(target_model)

    task_desc = getattr(task, "desc", None)
    if task_desc is not None:
        task_desc = str(task_desc)

    return workload_key, target_kind, target_model, task_desc


def _extract_measure_task_signature_from_payload(
    payload: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    inp_payload = payload["i"][0]
    workload_key = str(inp_payload[0]) if len(inp_payload) > 0 else None
    target_desc = str(inp_payload[1]) if len(inp_payload) > 1 else ""
    target_kind = target_desc.split(" ", 1)[0] if target_desc else None
    target_model = "unknown"
    task_desc = ""
    return workload_key, target_kind, target_model, task_desc


def _transform_cost_for_training(cost: Optional[float]) -> Optional[float]:
    if cost is None:
        return None
    try:
        value = float(cost)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value in (0.0, 1e10):
        return None
    if value <= 0.0:
        return None
    return float(-math.log(value))


def _extract_measure_cost(result) -> Optional[float]:
    try:
        error_no = int(result.error_no)
    except (TypeError, ValueError):
        error_no = 0

    if error_no != 0:
        return None

    costs = [float(x) for x in result.costs]
    if not costs:
        return None
    mean_cost = float(sum(costs) / len(costs))
    return _transform_cost_for_training(mean_cost)


def _extract_measure_cost_from_payload(payload: dict) -> Optional[float]:
    result = payload.get("r")
    if not isinstance(result, list) or len(result) < 2:
        return None

    error_no = _maybe_int(result[1])
    if error_no is None or error_no != 0:
        return None

    costs = result[0]
    if not isinstance(costs, list) or not costs:
        return None

    try:
        mean_cost = float(sum(float(x) for x in costs) / len(costs))
    except (TypeError, ValueError):
        return None
    return _transform_cost_for_training(mean_cost)


def _load_custom_json_sample(path: Path, payload: dict, record_index: Optional[int] = None) -> JsonSampleRecord:
    task_index, workload_key, target_kind, target_model, task_desc = _extract_task_signature(payload)
    sketch_index = _maybe_int((payload.get("meta", {}) or {}).get("sketch_index"))
    if sketch_index is None:
        sketch_index = 0

    cost = payload.get("cost")
    if cost is None:
        cost = payload.get("cost_target")
    cost = _transform_cost_for_training(cost)

    params = _extract_param_dict(payload)

    return JsonSampleRecord(
        sample_id=_build_sample_id(
            path,
            workload_key,
            target_kind,
            sketch_index,
            task_index,
            record_index=record_index,
        ),
        json_path=str(path),
        sketch_index=int(sketch_index),
        params=params,
        cost=cost,
        raw=payload,
        workload_key=workload_key,
        target_kind=target_kind,
        target_model=target_model,
        task_desc=task_desc,
        task_index=task_index,
        record_index=record_index,
        param_signature=tuple(params.keys()),
    )


def _load_measure_record_sample(path: Path, line: str, record_index: int) -> JsonSampleRecord:
    payload = json.loads(line)

    try:
        params, param_signature = _extract_measure_record_params_from_payload(payload)
        workload_key, target_kind, target_model, task_desc = (
            _extract_measure_task_signature_from_payload(payload)
        )
        cost = _extract_measure_cost_from_payload(payload)
    except (KeyError, TypeError, ValueError, IndexError):
        from tvm.auto_scheduler.measure_record import load_record_from_string

        inp, res = load_record_from_string(line)
        params, param_signature = _extract_measure_record_params(inp.state)
        workload_key, target_kind, target_model, task_desc = _extract_search_task_signature(inp.task)
        cost = _extract_measure_cost(res)

    return JsonSampleRecord(
        sample_id=_build_sample_id(
            path,
            workload_key,
            target_kind,
            0,
            task_index=None,
            record_index=record_index,
        ),
        json_path=str(path),
        sketch_index=0,
        params=params,
        cost=cost,
        raw=payload,
        workload_key=workload_key,
        target_kind=target_kind,
        target_model=target_model,
        task_desc=task_desc,
        task_index=None,
        record_index=record_index,
        param_signature=param_signature,
    )


def load_json_sample(path: str | Path) -> JsonSampleRecord:
    path = Path(path)
    samples = load_json_samples(path)
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples[0]


def load_json_samples(path: str | Path) -> List[JsonSampleRecord]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if "i" in payload and "r" in payload:
            return [_load_measure_record_sample(path, text.strip(), 0)]
        return [_load_custom_json_sample(path, payload)]

    samples: List[JsonSampleRecord] = []
    for record_index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "i" in payload and "r" in payload:
            samples.append(_load_measure_record_sample(path, line, record_index))
        else:
            samples.append(_load_custom_json_sample(path, payload, record_index=record_index))

    return samples


# -----------------------------------------------------------------------------
# Dataset split helper
# -----------------------------------------------------------------------------


def split_records(
    records: Sequence[JsonSampleRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[JsonSampleRecord], List[JsonSampleRecord], List[JsonSampleRecord]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    shuffled = [records[i] for i in indices]

    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio))) if n >= 2 else n
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = max(0, n - n_train - n_val)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]

    if val_ratio > 0.0 and not val and len(train) > 1:
        val = [train.pop()]
    if test_ratio > 0.0 and not test and len(train) > 1:
        test = [train.pop()]
    return train, val, test


# -----------------------------------------------------------------------------
# Generator reconstruction
# -----------------------------------------------------------------------------


class GeneratorRegistry:
    """
    역할
    - JSON payload에서 task identity를 해석해 실제 ScheduleGenerator를 복원한다.
    - task/select → concrete sketch 확보 → symbolic state → constraint generator 경로를 감싼다.

    입력
    - network_info_folder
    - JSON의 workload_key / target_kind / sketch_index
    - (fallback) task_index / sketch_index

    출력
    - modules.schedule_generator.ScheduleGenerator

    핵심 가정
    - workload_key + target_kind가 dataset에서 task를 사실상 유일하게 식별한다.
    - sketch_index는 동일 task에 대해 SketchPolicy.generate_concrete_sketches() 순서와 일치한다.

    실패 지점
    - network_info_folder mismatch
    - same workload_key,target_kind duplicated in dataset
    - sketch ordering mismatch
    - TVM custom runtime / global func 누락
    """

    def __init__(self, network_info_folder: str):
        self.network_info_folder = network_info_folder
        self._tasks = None
        self._tasks_by_index: Dict[int, object] = {}
        self._tasks_by_signature: Dict[Tuple[str, str], object] = {}
        self._task_index_by_signature: Dict[Tuple[str, str], int] = {}
        self._sketch_cache: Dict[Tuple[str, str], list] = {}
        self._generator_cache: Dict[Tuple[str, str, int], object] = {}
        self._sketch_index_by_param_signature: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}

    def _load_tasks(self):
        if self._tasks is not None:
            return self._tasks

        from modules.task_paths import load_and_register_tasks

        tasks = load_and_register_tasks(self.network_info_folder)
        self._tasks = tasks
        self._tasks_by_index = {idx: task for idx, task in enumerate(tasks)}

        by_sig: Dict[Tuple[str, str], List[int]] = {}
        for idx, task in enumerate(tasks):
            sig = (str(task.workload_key), str(task.target.kind))
            by_sig.setdefault(sig, []).append(idx)

        duplicates = {sig: ids for sig, ids in by_sig.items() if len(ids) > 1}
        if duplicates:
            example_sig, ids = next(iter(duplicates.items()))
            raise RuntimeError(
                "Task signature is not unique. "
                f"signature={example_sig}, task_indices={ids}"
            )

        self._tasks_by_signature = {
            sig: self._tasks_by_index[ids[0]]
            for sig, ids in by_sig.items()
        }
        self._task_index_by_signature = {
            sig: ids[0]
            for sig, ids in by_sig.items()
        }
        return tasks

    @staticmethod
    def _task_signature(task) -> Tuple[str, str]:
        return str(task.workload_key), str(task.target.kind)

    def _resolve_task(
        self,
        *,
        workload_key: Optional[str] = None,
        target_kind: Optional[str] = None,
        task_index: Optional[int] = None,
    ):
        self._load_tasks()

        if workload_key is not None and target_kind is not None:
            sig = (str(workload_key), str(target_kind))
            task = self._tasks_by_signature.get(sig)
            if task is None:
                raise KeyError(
                    "Task not found by workload signature: "
                    f"workload_key={workload_key}, target_kind={target_kind}"
                )
            return task

        if task_index is not None:
            task = self._tasks_by_index.get(int(task_index))
            if task is None:
                raise KeyError(f"Task index not found: task_index={task_index}")
            return task

        raise ValueError(
            "Need either (workload_key, target_kind) or task_index to resolve task"
        )

    def _get_sketches_for_task(self, task):
        sig = self._task_signature(task)
        cached = self._sketch_cache.get(sig)
        if cached is not None:
            return cached

        from tvm.auto_scheduler import SketchPolicy

        policy = SketchPolicy(task, params={"sample_init_no_invalid": 1}, verbose=False)
        sketches = list(policy.generate_concrete_sketches())
        self._sketch_cache[sig] = sketches
        return sketches

    def get_task_index(
        self,
        *,
        workload_key: Optional[str] = None,
        target_kind: Optional[str] = None,
        task_index: Optional[int] = None,
    ) -> int:
        self._load_tasks()

        if task_index is not None:
            normalized = int(task_index)
            if normalized not in self._tasks_by_index:
                raise KeyError(f"Task index not found: task_index={task_index}")
            return normalized

        if workload_key is not None and target_kind is not None:
            sig = (str(workload_key), str(target_kind))
            resolved = self._task_index_by_signature.get(sig)
            if resolved is None:
                raise KeyError(
                    "Task not found by workload signature: "
                    f"workload_key={workload_key}, target_kind={target_kind}"
                )
            return int(resolved)

        raise ValueError(
            "Need either (workload_key, target_kind) or task_index to resolve task index"
        )

    def get_generator(
        self,
        *,
        sketch_index: int,
        workload_key: Optional[str] = None,
        target_kind: Optional[str] = None,
        task_index: Optional[int] = None,
    ):
        task = self._resolve_task(
            workload_key=workload_key,
            target_kind=target_kind,
            task_index=task_index,
        )
        task_workload_key, task_target_kind = self._task_signature(task)
        cache_key = (task_workload_key, task_target_kind, int(sketch_index))
        cached = self._generator_cache.get(cache_key)
        if cached is not None:
            return cached

        from modules.schedule_generator import ScheduleGenerator

        sketches = self._get_sketches_for_task(task)
        try:
            state = sketches[int(sketch_index)]
        except IndexError as err:
            raise IndexError(
                f"sketch_index={sketch_index} out of range for "
                f"workload_key={task_workload_key}, target_kind={task_target_kind}; "
                f"available={len(sketches)}"
            ) from err

        gen = ScheduleGenerator.from_task_state(
            task,
            state,
            enabled_constraints=_runtime_enabled_constraints(),
        )
        self._generator_cache[cache_key] = gen
        return gen

    def get_generator_from_record(self, record: JsonSampleRecord):
        task = self._resolve_task(
            workload_key=record.workload_key,
            target_kind=record.target_kind,
            task_index=record.task_index,
        )

        if record.param_signature:
            task_workload_key, task_target_kind = self._task_signature(task)
            normalized_signature = _normalize_generator_signature(record.param_signature)
            cache_key = (task_workload_key, task_target_kind, normalized_signature)
            resolved_index = self._sketch_index_by_param_signature.get(cache_key)
            if resolved_index is None:
                matches = []
                for sketch_index, _ in enumerate(self._get_sketches_for_task(task)):
                    gen = self.get_generator(
                        workload_key=task_workload_key,
                        target_kind=task_target_kind,
                        sketch_index=sketch_index,
                    )
                    if tuple(gen.s.sym_map.keys()) == normalized_signature:
                        matches.append(sketch_index)

                if len(matches) == 1:
                    resolved_index = matches[0]
                    self._sketch_index_by_param_signature[cache_key] = resolved_index
                elif len(matches) > 1:
                    raise RuntimeError(
                        "Multiple sketches matched the same param signature. "
                        f"workload_key={task_workload_key}, target_kind={task_target_kind}, "
                        f"matches={matches}"
                    )

            if resolved_index is not None:
                record.sketch_index = int(resolved_index)
                record.sample_id = _build_sample_id(
                    Path(record.json_path),
                    record.workload_key,
                    record.target_kind,
                    record.sketch_index,
                    record.task_index,
                    record_index=record.record_index,
                )

        gen = self.get_generator(
            workload_key=record.workload_key,
            target_kind=record.target_kind,
            task_index=record.task_index,
            sketch_index=record.sketch_index,
        )
        _augment_record_budget_params(record, gen)
        return gen

    def get_generator_from_payload(self, payload: dict):
        task_index, workload_key, target_kind, _, _ = _extract_task_signature(payload)
        sketch_index = _maybe_int((payload.get("meta", {}) or {}).get("sketch_index"))
        if sketch_index is None:
            sketch_index = 0
        return self.get_generator(
            workload_key=workload_key,
            target_kind=target_kind,
            task_index=task_index,
            sketch_index=sketch_index,
        )

    def build_oracle_from_record(self, record: JsonSampleRecord) -> "LegalPrefixOracle":
        gen = self.get_generator_from_record(record)
        overrides = _record_split_extent_overrides(record)
        return LegalPrefixOracle(gen, extent_overrides=overrides)

    def build_oracle_from_payload(self, payload: dict) -> "LegalPrefixOracle":
        return LegalPrefixOracle(self.get_generator_from_payload(payload))

    def build_oracle(
        self,
        task_index: Optional[int] = None,
        sketch_index: int = 0,
        *,
        workload_key: Optional[str] = None,
        target_kind: Optional[str] = None,
    ) -> "LegalPrefixOracle":
        return LegalPrefixOracle(
            self.get_generator(
                workload_key=workload_key,
                target_kind=target_kind,
                task_index=task_index,
                sketch_index=sketch_index,
            )
        )


# -----------------------------------------------------------------------------
# Prefix legality adapter
# -----------------------------------------------------------------------------


class LegalPrefixOracle:
    """
    역할
    - prefix 기반 legality 상태를 한 샘플 단위로 관리한다.
    - candidate query → assignment update → 다음 domain 반영을 순차적으로 수행한다.

    입력
    - ScheduleGenerator
    - prefix assignment

    출력
    - 현재 변수의 valid candidate 집합
    - assignment 후 전파된 도메인 상태

    핵심 가정
    - generator.get_param_candidates / propagate_param_assignment가 prefix를 기준으로
      내부 sym_map을 materialize하고 복원한다.

    실패 지점
    - invalid ground-truth assignment
    - upstream observability format change
    """

    def __init__(self, generator, extent_overrides=None):
        self.generator = generator
        self._extent_overrides = (
            {int(k): int(v) for k, v in extent_overrides.items()}
            if extent_overrides else None
        )
        self._extent_fingerprint = (
            tuple(sorted(self._extent_overrides.items())) if self._extent_overrides else ()
        )
        _, domains, group_remaining, _ = generator.param_sampler._initialize_unique_search_base_state(
            generator._var_order,
            extent_overrides=self._extent_overrides,
        )
        self.assignment: Dict[str, int] = {}
        self._domains = generator.param_sampler._copy_domains(domains)
        self._group_remaining = generator.param_sampler._copy_group_remaining(group_remaining)
        self._budget_remaining: Dict[str, int] = {}
        self.last_report: Optional[dict] = None
        root_snapshot = (
            dict(self.assignment),
            self.generator.param_sampler._copy_domains(self._domains),
            self.generator.param_sampler._copy_group_remaining(self._group_remaining),
            self.generator.param_sampler._copy_budget_remaining(self._budget_remaining),
            dict(self.generator.s.sym_map),
        )
        if not hasattr(self.generator, "_lpm_candidate_cache"):
            self.generator._lpm_candidate_cache = {}
        if not hasattr(self.generator, "_lpm_mask_cache"):
            self.generator._lpm_mask_cache = {}
        root_key = (self._extent_fingerprint, tuple())
        if not hasattr(self.generator, "_lpm_prefix_state_cache"):
            self.generator._lpm_prefix_state_cache = {root_key: root_snapshot}
        elif root_key not in self.generator._lpm_prefix_state_cache:
            self.generator._lpm_prefix_state_cache[root_key] = root_snapshot

        cached_root = self.generator._lpm_prefix_state_cache[root_key]
        self.assignment.clear()
        self.assignment.update(cached_root[0])
        self._domains = self.generator.param_sampler._copy_domains(cached_root[1])
        self._group_remaining = self.generator.param_sampler._copy_group_remaining(cached_root[2])
        self._budget_remaining = self.generator.param_sampler._copy_budget_remaining(cached_root[3])
        self._sym_map = dict(cached_root[4])
        self.generator.param_sampler._restore_sym_map(self._sym_map)

    def _activate_state(self) -> None:
        self.generator.param_sampler._restore_sym_map(self._sym_map)

    def mask_cache_key(self, prefix_tuple, var_name) -> tuple:
        return (self._extent_fingerprint, tuple(prefix_tuple), str(var_name))

    def prefix_state_key(self, prefix_tuple) -> tuple:
        return (self._extent_fingerprint, tuple(prefix_tuple))

    def param_order(self) -> List[str]:
        return list(self.generator.get_full_var_order_entries()["param_order"])

    def candidate_values(self, var_name: str) -> List[int]:
        self._activate_state()
        cache_key = (self._extent_fingerprint, tuple(self.assignment.items()), str(var_name))
        cached = self.generator._lpm_candidate_cache.get(cache_key)
        if cached is not None:
            candidates = list(cached)
        else:
            candidates = self.generator.param_sampler._get_search_candidates(
                var_name,
                self._domains,
                self._group_remaining,
                self._budget_remaining,
                self.assignment,
                extent_overrides=self._extent_overrides,
            )
            self.generator._lpm_candidate_cache[cache_key] = tuple(int(v) for v in candidates)
        candidates = list(sorted(dict.fromkeys(int(v) for v in candidates)))
        self.last_report = {
            "query": {"param_name": var_name},
            "assignment": {"params": dict(self.assignment)},
            "candidates": list(candidates),
        }
        return candidates

    def assign(self, var_name: str, value: int) -> dict:
        value = int(value)
        self._activate_state()
        self.generator.param_sampler._apply_search_assignment(
            var_name,
            value,
            self.assignment,
            self._domains,
            self._group_remaining,
            self._budget_remaining,
            extent_overrides=self._extent_overrides,
        )
        self._sym_map = dict(self.generator.s.sym_map)
        self.last_report = {
            "query": {"param_name": var_name, "param_value": value},
            "assignment": {"params": dict(self.assignment)},
        }
        self.generator._lpm_prefix_state_cache[(self._extent_fingerprint, tuple(self.assignment.items()))] = (
            dict(self.assignment),
            self.generator.param_sampler._copy_domains(self._domains),
            self.generator.param_sampler._copy_group_remaining(self._group_remaining),
            self.generator.param_sampler._copy_budget_remaining(self._budget_remaining),
            dict(self.generator.s.sym_map),
        )
        return self.last_report

    def validate_assignment(
        self,
        ordered_names: Sequence[str],
        ordered_values: Sequence[int],
    ) -> bool:
        self.assignment.clear()
        for name, value in zip(ordered_names, ordered_values):
            try:
                candidates = self.candidate_values(name)
            except Exception:  # pylint: disable=broad-except
                return False
            if int(value) not in candidates:
                return False
            self.assign(name, int(value))
        return True

    def final_violations(self) -> List[str]:
        return list(self.generator.check_all_hybrid(self.assignment))
