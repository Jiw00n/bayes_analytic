from __future__ import annotations

import gc
import json
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from tvm import auto_scheduler

from .adapter import (
    GeneratorRegistry,
    JsonSampleRecord,
    LegalPrefixOracle,
    _parse_shapes_from_workload_key,
    cost_label_to_raw,
    cost_raw_to_label,
    load_json_sample,
    load_json_samples,
)
from .dataset import (
    budget_enabled,
    get_model_param_order,
)
from .inference import SamplingOptions, _sample_token_from_logits
from .model import LatentParamVAE
from .shape_semantics import (
    flatten_labels,
    semantic_labels_for_task,
)
from .recon_predict_gp import make_task_sym_map_key
from .tokenizer import ParamTokenizer
from modules.task_paths import clean_name, get_measure_record_filename
from result_csv_utils import (
    append_result_rows,
    extract_true_mean_cost,
    load_existing_deterministic_sample_ids,
    load_existing_random_seeds,
    make_sym_map_key,
    resolve_results_csv_path,
)


# -----------------------------------------------------------------------------
# Measurement helpers
# -----------------------------------------------------------------------------

RUN_TIMEOUT = 5
NUMBER = 1
VERBOSE = 1



@dataclass
class MeasurementSession:
    task: Any
    output_path: str
    measurer: Any
    policy: Any


def _task_repeat(task: Any) -> int:
    """task FLOPS 기반 반복 횟수 선택."""
    # measure_programs.py와 동일 정책을 사용한다.
    return 3


def _make_measurer(task: Any, log_filename: str) -> Any:
    """task용 ProgramMeasurer를 만든다."""
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=RUN_TIMEOUT,
        repeat=_task_repeat(task),
        number=NUMBER,
        enable_cpu_cache_flush=False,
    )
    return auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=VERBOSE,
    )


def _build_measurement_session(
    task: Any,
    measure_output_dir: Optional[str] = None,
    *,
    output_path: Optional[str] = None,
) -> MeasurementSession:
    if output_path is None:
        output_path = str(
            get_measure_record_filename(task, task.target, output_dir=measure_output_dir)
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return MeasurementSession(
        task=task,
        output_path=output_path,
        measurer=_make_measurer(task, output_path),
        policy=auto_scheduler.search_policy.EmptyPolicy(task),
    )


def measure_candidate(
    *,
    session: MeasurementSession,
    result: Any,
    meta: Dict[str, Any],
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """단일 MeasureResult를 요약 딕셔너리로 변환한다.

    ``mean_cost`` 필드는 ``cost_target`` 공간의 라벨 값 (훈련 라벨과 같은
    스케일). 원시 초단위 값은 ``costs`` 필드에 보존된다."""
    error_no = int(result.error_no)
    costs = [float(x) for x in result.costs]
    raw_mean_cost = float(sum(costs) / len(costs)) if costs else None
    usable_measurement = error_no == int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    error_msg = str(result.error_msg)
    mean_cost_label = cost_raw_to_label(
        raw_mean_cost, cost_target, task_min_cost=task_min_cost
    )
    return {
        "ok": True,
        "usable_measurement": usable_measurement,
        "error_no": error_no,
        "error_msg": error_msg or None,
        "costs": costs,
        "mean_cost": mean_cost_label,
        "all_cost": float(result.all_cost),
        "timestamp": float(result.timestamp),
        "measure_record_path": session.output_path,
        "meta": dict(meta),
    }


def measure_candidates_batch(
    *,
    session: MeasurementSession,
    task: Any,
    states: List[Any],
    metas: List[Dict[str, Any]],
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """여러 candidate를 한 번의 measurer.measure 호출로 측정한다."""
    if not states:
        return []

    try:
        measure_inputs = [auto_scheduler.MeasureInput(task, state) for state in states]
        results = session.measurer.measure(task, session.policy, measure_inputs)
    except Exception as err:  # pylint: disable=broad-except
        error_text = f"{type(err).__name__}: {err}"
        return [
            {
                "ok": False,
                "stage": "measure_records",
                "error": error_text,
                "measure_record_path": session.output_path,
                "meta": dict(meta),
            }
            for meta in metas
        ]

    if len(results) != len(states):
        error_text = (
            "measurement returned mismatched result count: "
            f"expected={len(states)} actual={len(results)}"
        )
        return [
            {
                "ok": False,
                "stage": "measure_records",
                "error": error_text,
                "measure_record_path": session.output_path,
                "meta": dict(meta),
            }
            for meta in metas
        ]

    return [
        measure_candidate(
            result=result,
            meta=meta,
            session=session,
            cost_target=cost_target,
            task_min_cost=task_min_cost,
        )
        for result, meta in zip(results, metas)
    ]


def log_grouped_candidate_result(
    grouped_record: Dict[str, Any],
    *,
    reencode_label: str = "re_pred",
    show_neg_log: bool = False,
) -> None:
    record = grouped_record["record"]
    alphas = grouped_record["alphas"]

    alpha_text = json.dumps(alphas, ensure_ascii=False)

    if record.final_violations:
        print(
            f"[latent-walk] alphas={alpha_text} status=violated "
            f"violations={len(record.final_violations)}"
        )
        return

    if not record.state_build_ok:
        print(
            f"[latent-walk] alphas={alpha_text} status=state_build_failed "
            f"error={record.state_build_error}"
        )
        return

    measurement = record.measurement or {}
    if not measurement.get("ok"):
        print(
            f"[latent-walk] alphas={alpha_text} status=measure_failed "
            f"error={measurement.get('error')}"
        )
        return

    if not measurement.get("usable_measurement"):
        print(
            f"[latent-walk] alphas={alpha_text} status=measure_error "
            f"error_no={measurement.get('error_no')} "
            f"error_msg={measurement.get('error_msg')}"
        )
        return

    raw_costs = measurement.get("costs") or []
    measured_raw = (
        float(sum(float(c) for c in raw_costs) / len(raw_costs))
        if raw_costs
        else None
    )
    measured_text = "n/a" if measured_raw is None else f"{measured_raw:.5g}"
    reencode_pred = record.reencode_cost_pred
    reencode_text = "n/a" if reencode_pred is None else f"{reencode_pred:.4f}"
    parts = [f"measured={measured_text}"]
    if show_neg_log:
        if measured_raw is not None and measured_raw > 0.0:
            neg_log_text = f"{-math.log(measured_raw):.4f}"
        else:
            neg_log_text = "n/a"
        parts.append(f"-log={neg_log_text}")
    parts.append(f"{reencode_label}={reencode_text}")
    parts.append(alpha_text)
    print(", ".join(parts))


def _family_from_record(record: JsonSampleRecord) -> Optional[str]:
    """Family is ``Path(record.json_path).parents[1].name`` when the dataset
    directory layout is ``.../{family}/{target}/{task_index}_*.json``."""
    json_path = getattr(record, "json_path", None)
    if not json_path:
        return None
    p = Path(json_path)
    if len(p.parents) < 2:
        return None
    return p.parents[1].name


def _resolve_walk_measure_record_path(
    *,
    record: JsonSampleRecord,
    task: Any,
    output: Optional[str | Path],
) -> Optional[str]:
    """Compose the per-task measurement record path for latent walk:

        ``{output}/{family}/measure_records/{task_index}_{clean_name}.json``

    where ``output`` is the configured checkpoint dir (e.g.
    ``checkpoints_all``). All tasks for the same family share the single
    ``{family}/measure_records/`` directory, distinguished by the
    ``{task_index}_`` filename prefix.

    ``task_index`` falls back to the leading digits of the source JSON's
    filename stem when ``record.task_index`` is missing (the common case for
    JSON payloads that don't carry ``meta.task_index`` / ``task.task_index``).
    """
    if output is None:
        return None
    import re

    target = getattr(task, "target", None)
    target_kind = str(target.kind) if target is not None else ""
    base_key = clean_name((task.workload_key, target_kind))
    family = _family_from_record(record)
    task_index: Optional[int] = getattr(record, "task_index", None)
    if task_index is None:
        json_path = getattr(record, "json_path", None)
        if json_path:
            m = re.match(r"^(\d+)", Path(json_path).stem)
            if m:
                try:
                    task_index = int(m.group(1))
                except ValueError:
                    task_index = None
    output_root = Path(output)
    if family:
        output_root = output_root / family
    output_dir = output_root / "measure_records"
    if task_index is not None:
        filename = f"{int(task_index)}_{base_key}.json"
    else:
        filename = f"{base_key}.json"
    return str(output_dir / clean_name(filename))


def _select_record_from_path(
    record_json_path: str | Path,
    *,
    best_cost: bool = False,
) -> JsonSampleRecord:
    if not best_cost:
        return load_json_sample(record_json_path)

    records = load_json_samples(record_json_path)
    if not records:
        raise ValueError(f"No samples found in {record_json_path}")

    records_with_cost = [
        record for record in records
        if record.cost is not None and math.isfinite(float(record.cost))
    ]
    if not records_with_cost:
        raise ValueError(
            f"--best-cost requested but no finite cost records were found in {record_json_path}"
        )
    return max(records_with_cost, key=lambda record: float(record.cost))


def _release_measurement_environment(
    bundle: LoadedBundle,
    *,
    generator: Any | None = None,
    keep_bundle: bool = False,
) -> None:
    if not keep_bundle:
        registry = getattr(bundle, "registry", None)
        if registry is not None:
            for attr_name in (
                "_generator_cache",
                "_sketch_index_by_param_signature",
                "_sketch_cache",
                "_tasks_by_index",
                "_tasks_by_signature",
            ):
                cache = getattr(registry, attr_name, None)
                if isinstance(cache, dict):
                    cache.clear()
            if hasattr(registry, "_tasks"):
                registry._tasks = None

    if generator is not None:
        for attr_name in (
            "_concrete_final_cache",
            "_concrete_partial_cache",
            "_params_to_state_cache",
            "_check_prefix_cache",
            "_materialize_cache",
            "_assignment_validation_cache",
            "_full_check_cache",
            "_lpm_mask_cache",
            "_lpm_prefix_state_cache",
        ):
            cache = getattr(generator, attr_name, None)
            if isinstance(cache, dict):
                cache.clear()

    if not keep_bundle:
        bundle.model = None
        bundle.tokenizer = None
        bundle.cost_weight = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


@dataclass
class LoadedBundle:
    checkpoint_payload: Dict[str, Any]
    model: LatentParamVAE
    tokenizer: ParamTokenizer
    registry: GeneratorRegistry
    cost_weight: Optional[torch.Tensor]
    cost_bias: float
    cost_source: str
    device: torch.device
    recon_predictor: Optional[Any] = None  # GPReconPredictor; swaps in for cost_head in predict_recon_score
    reencode_predictor: Optional[Any] = None  # ReEncodePredictor; used by predict_reencode_score
    reencode_predictor_name: str = "cost_head"
    # Space metadata for the cost ridge (``cost_weight``/``cost_bias``). The
    # raw ``z @ weight + bias`` output is in ``ridge_fit_target`` space; it is
    # converted to ``ridge_output_target`` (which callers assume = cost_target)
    # by ``predict_score`` using ``ridge_task_min_cost`` when needed.
    ridge_fit_target: str = "neg_log"
    ridge_output_target: str = "neg_log"
    ridge_task_min_cost: Optional[float] = None


@dataclass
class WalkRecord:
    step_index: int
    alpha: float
    predicted_score: float
    z: List[float]
    params: Dict[str, int]
    sym_map: Dict[str, int]
    final_violations: List[str]
    state_build_ok: bool
    state_build_error: Optional[str]
    measurement: Any
    recon_predict_cost: Optional[float] = None
    recon_predict_std: Optional[float] = None
    reencode_cost_pred: Optional[float] = None


@dataclass
class DecodeRecord:
    params: Dict[str, int]
    sym_map: Dict[str, int]
    final_violations: List[str]


@dataclass
class StartZContext:
    z0: torch.Tensor
    generator: Any
    ordered_names: List[str]
    ordered_values: List[int]
    encode_deterministic: Optional[bool]
    encoded_z: Optional[List[float]]



def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)



def _make_model_cfg(cfg_payload: Dict[str, Any]) -> Any:
    return SimpleNamespace(**cfg_payload)



def load_bundle(
    checkpoint_path: str | Path,
    *,
    network_info_folder: Optional[str] = None,
    device: str = "cuda",
    use_latent_gradient: bool = False,
) -> LoadedBundle:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = ParamTokenizer.from_state_dict(payload["tokenizer"])
    model_cfg = _make_model_cfg(payload["config"]["model"])
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=model_cfg,
    )
    model.load_state_dict(payload["model_state"])

    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    model.eval()

    latent_cost_ridge = payload.get("latent_cost_ridge")
    cost_weight = None
    cost_bias = 0.0
    cost_source = "missing_cost_vector"
    ridge_fit_target = "neg_log"
    ridge_output_target = "neg_log"
    ridge_task_min_cost: Optional[float] = None
    if latent_cost_ridge is not None and "weight" in latent_cost_ridge:
        cost_weight = latent_cost_ridge["weight"].detach().to(
            dtype=torch.float32,
            device=torch_device,
        )
        cost_bias = float(latent_cost_ridge.get("bias", 0.0))
        cost_source = "latent_cost_ridge"
        ridge_fit_target = str(latent_cost_ridge.get("target_name", "neg_log"))
        ridge_output_target = str(latent_cost_ridge.get("cost_target", ridge_fit_target))
        tmc = latent_cost_ridge.get("task_min_cost")
        ridge_task_min_cost = float(tmc) if tmc is not None else None
    elif use_latent_gradient:
        cost_source = "cost_head_gradient"

    if network_info_folder is None:
        network_info_folder = payload["config"]["data"]["network_info_folder"]

    generator_cfg = payload.get("config", {}).get("generator", {}) or {}
    registry = GeneratorRegistry(
        network_info_folder,
        hw_param=generator_cfg.get("hw_param") or None,
        disable_constraint=generator_cfg.get("disable_constraint") or None,
    )
    return LoadedBundle(
        checkpoint_payload=payload,
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        cost_weight=cost_weight,
        cost_bias=cost_bias,
        cost_source=cost_source,
        device=torch_device,
        ridge_fit_target=ridge_fit_target,
        ridge_output_target=ridge_output_target,
        ridge_task_min_cost=ridge_task_min_cost,
    )


def make_bundle(
    *,
    model: LatentParamVAE,
    tokenizer: ParamTokenizer,
    registry: GeneratorRegistry,
    config_payload: Dict[str, Any],
    latent_cost_ridge: Optional[Dict[str, Any]] = None,
    device: torch.device,
    use_latent_gradient: bool = False,
    timestamp: Optional[str] = None,
    recon_predictor: Optional[Any] = None,
    reencode_predictor: Optional[Any] = None,
    reencode_predictor_name: str = "cost_head",
) -> LoadedBundle:
    """Construct a LoadedBundle from already-loaded in-memory objects.

    Avoids the disk round-trip that load_bundle incurs: no torch.load, no model
    reconstruction, no GeneratorRegistry rebuild. Intended for the training loop
    so periodic latent walks reuse the live training model + cached registry.
    """
    cost_weight = None
    cost_bias = 0.0
    cost_source = "missing_cost_vector"
    ridge_fit_target = "neg_log"
    ridge_output_target = "neg_log"
    ridge_task_min_cost: Optional[float] = None
    if latent_cost_ridge is not None and "weight" in latent_cost_ridge:
        cost_weight = latent_cost_ridge["weight"].detach().to(
            dtype=torch.float32,
            device=device,
        )
        cost_bias = float(latent_cost_ridge.get("bias", 0.0))
        cost_source = "latent_cost_ridge"
        ridge_fit_target = str(latent_cost_ridge.get("target_name", "neg_log"))
        ridge_output_target = str(latent_cost_ridge.get("cost_target", ridge_fit_target))
        tmc = latent_cost_ridge.get("task_min_cost")
        ridge_task_min_cost = float(tmc) if tmc is not None else None
    elif use_latent_gradient:
        cost_source = "cost_head_gradient"

    payload: Dict[str, Any] = {"config": config_payload}
    if timestamp is not None:
        payload["timestamp"] = timestamp

    return LoadedBundle(
        checkpoint_payload=payload,
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        cost_weight=cost_weight,
        cost_bias=cost_bias,
        cost_source=cost_source,
        device=device,
        recon_predictor=recon_predictor,
        reencode_predictor=reencode_predictor,
        reencode_predictor_name=reencode_predictor_name,
        ridge_fit_target=ridge_fit_target,
        ridge_output_target=ridge_output_target,
        ridge_task_min_cost=ridge_task_min_cost,
    )


@torch.no_grad()
def prepare_record_context(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
) -> tuple[Any, List[str], List[int]]:
    gen = bundle.registry.get_generator_from_record(record)
    ordered_names = get_model_param_order(
        gen,
        include_budget=budget_enabled(bundle.checkpoint_payload.get("config", {})),
    )
    ordered_values = [int(record.params[name]) for name in ordered_names]
    return gen, ordered_names, ordered_values


@torch.no_grad()
def shape_prefix_for_record(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
) -> tuple[List[int], List[int]]:
    """Return (shape_token_ids, shape_var_ids) for the record's workload.

    Matches the encoder/decoder prefix layout produced at training time in
    dataset._build_prepared_sample.
    """
    shapes = record.shapes
    if shapes is None:
        shapes = _parse_shapes_from_workload_key(record.workload_key)
    if shapes is None:
        return [], []
    task = bundle.registry._resolve_task(
        workload_key=record.workload_key,
        target_kind=record.target_kind,
        task_index=record.task_index,
    )
    nested_labels = semantic_labels_for_task(task)
    if len(nested_labels) != len(shapes):
        raise ValueError(
            f"compute_dag tensor count ({len(nested_labels)}) does not match "
            f"parsed shapes ({len(shapes)}) for {record.sample_id}"
        )
    flat_labels = flatten_labels(nested_labels)
    shape_values: List[int] = []
    for shape in shapes:
        shape_values.extend(int(v) for v in shape)
    if len(flat_labels) != len(shape_values):
        raise ValueError(
            f"shape semantic label count mismatch for {record.sample_id}: "
            f"labels={len(flat_labels)} values={len(shape_values)}"
        )
    tokenizer = bundle.tokenizer
    shape_token_ids = [
        tokenizer.token_to_id.get(str(int(v)), tokenizer.unk_id)
        for v in shape_values
    ]
    shape_var_ids = [tokenizer.var_to_id[label] for label in flat_labels]
    return shape_token_ids, shape_var_ids


def _extent_token_enabled_for_bundle(bundle: "LoadedBundle") -> bool:
    cfg_payload = bundle.checkpoint_payload.get("config", {})
    if isinstance(cfg_payload, dict):
        data_payload = cfg_payload.get("data", {})
        if isinstance(data_payload, dict):
            return bool(data_payload.get("extent_token", False))
    return False


@torch.no_grad()
def extent_prefix_for_record(
    bundle: "LoadedBundle",
    record: JsonSampleRecord,
    gen: Optional[Any] = None,
) -> tuple[List[int], List[int]]:
    """Return ``(extent_token_ids, extent_var_ids)`` for the record.

    Mirrors the prefix laid out at training time when
    ``data.extent_token=True``: one token per SplitStep, in step-index order,
    using ``gen._sp_extents`` for the value and ``sp_extent_{step_idx}`` for
    the var label. Returns empty lists when extent_token is disabled.
    """
    if not _extent_token_enabled_for_bundle(bundle):
        return [], []
    if gen is None:
        gen = bundle.registry.get_generator_from_record(record)
    sp_extents: Dict[int, int] = dict(getattr(gen, "_sp_extents", {}))
    if not sp_extents:
        return [], []
    sorted_indices = sorted(int(idx) for idx in sp_extents.keys())
    tokenizer = bundle.tokenizer
    extent_token_ids = [
        tokenizer.token_to_id.get(str(int(sp_extents[idx])), tokenizer.unk_id)
        for idx in sorted_indices
    ]
    extent_var_ids = [
        tokenizer.var_to_id[f"sp_extent_{idx}"] for idx in sorted_indices
    ]
    return extent_token_ids, extent_var_ids


@torch.no_grad()
def encode_record_to_z(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, Any, List[str], List[int]]:
    gen, ordered_names, ordered_values = prepare_record_context(bundle, record)

    shape_token_ids, shape_var_ids = shape_prefix_for_record(bundle, record)
    extent_token_ids, extent_var_ids = extent_prefix_for_record(bundle, record, gen=gen)
    enc_token_list = (
        shape_token_ids
        + extent_token_ids
        + bundle.tokenizer.encode_values(ordered_names, ordered_values)
    )
    enc_var_list = (
        shape_var_ids
        + extent_var_ids
        + bundle.tokenizer.encode_var_names(ordered_names)
    )
    enc_ids = torch.tensor([enc_token_list], dtype=torch.long, device=bundle.device)
    enc_var_ids = torch.tensor([enc_var_list], dtype=torch.long, device=bundle.device)
    enc_pad = enc_ids.eq(bundle.tokenizer.pad_id)
    _, _, z, _ = bundle.model.encode(
        enc_ids,
        enc_var_ids,
        enc_pad,
        deterministic=deterministic,
    )
    return z[0].detach().clone(), gen, ordered_names, ordered_values


@torch.no_grad()
def sample_gaussian_z(
    bundle: LoadedBundle,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    z = torch.randn(
        (int(bundle.model.cfg.latent_dim),),
        generator=generator,
        dtype=torch.float32,
    )
    return z.to(device=bundle.device, dtype=torch.float32)


@torch.no_grad()
def resolve_start_z(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
    *,
    random_z: bool = False,
    seed: Optional[int] = None,
    deterministic_start: bool = False,
) -> StartZContext:
    gen, ordered_names, ordered_values = prepare_record_context(bundle, record)
    if random_z:
        return StartZContext(
            z0=sample_gaussian_z(bundle, seed=seed),
            generator=gen,
            ordered_names=ordered_names,
            ordered_values=ordered_values,
            encode_deterministic=None,
            encoded_z=None,
        )
    z0, _, _, _ = encode_record_to_z(bundle, record, deterministic=deterministic_start)
    return StartZContext(
        z0=z0,
        generator=gen,
        ordered_names=ordered_names,
        ordered_values=ordered_values,
        encode_deterministic=bool(deterministic_start),
        encoded_z=None if deterministic_start else z0.detach().cpu().tolist(),
    )



def _resolve_decoded_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    token_id: int,
    candidate_values: List[int],
) -> int:
    token = tokenizer.id_to_token[int(token_id)]
    value = tokenizer.token_to_value(var_name, token)
    if value is not None:
        return int(value)
    if not candidate_values:
        raise RuntimeError(f"No legal candidates for {var_name}")
    return int(candidate_values[0])


@torch.no_grad()
def greedy_decode_from_zs_batch(
    bundle: LoadedBundle,
    oracles: List[LegalPrefixOracle],
    ordered_names: List[str],
    zs: torch.Tensor,
    *,
    sampling_options: Optional[SamplingOptions] = None,
    rng: Optional[torch.Generator] = None,
    shape_token_ids: Optional[List[int]] = None,
    shape_var_ids: Optional[List[int]] = None,
    extent_token_ids: Optional[List[int]] = None,
    extent_var_ids: Optional[List[int]] = None,
) -> List[DecodeRecord]:
    """Decode a batch of latent vectors in a single forward pass per token
    position.

    Each ``z`` gets its own ``oracle`` (constraint state) and its own
    candidate mask at every position, but all ``z``'s share the same
    decoder-input prefix length at any given step (param order is fixed), so
    we can batch the transformer ``decode`` call across z's. Per-z work that
    can't be batched — oracle.candidate_values, mask construction, sampling,
    oracle.assign — still happens in a Python loop, but the heavy transformer
    forward is paid once per step instead of B times.
    """
    options = sampling_options or SamplingOptions()
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    batch_size = len(oracles)
    if batch_size == 0:
        return []
    if zs.shape[0] != batch_size:
        raise ValueError(
            f"zs batch dim ({zs.shape[0]}) must match oracle count ({batch_size})"
        )

    zs = zs.to(device=device, dtype=torch.float32).reshape(batch_size, -1)
    memory = model.latent_to_memory(zs).view(
        batch_size, model.cfg.latent_token_count, model.cfg.embed_dim
    )

    shape_token_list = list(shape_token_ids or [])
    shape_var_list = list(shape_var_ids or [])
    if len(shape_token_list) != len(shape_var_list):
        raise ValueError(
            f"shape_token_ids ({len(shape_token_list)}) and shape_var_ids "
            f"({len(shape_var_list)}) length mismatch"
        )
    extent_token_list = list(extent_token_ids or [])
    extent_var_list = list(extent_var_ids or [])
    if len(extent_token_list) != len(extent_var_list):
        raise ValueError(
            f"extent_token_ids ({len(extent_token_list)}) and extent_var_ids "
            f"({len(extent_var_list)}) length mismatch"
        )
    prefix_token_list = shape_token_list + extent_token_list
    prefix_var_list = shape_var_list + extent_var_list
    param_var_ids = [tokenizer.var_to_id[name] for name in ordered_names]
    full_var_ids = prefix_var_list + param_var_ids

    base_decoder_ids: List[int] = list(prefix_token_list) + [tokenizer.param_start_id]
    decoder_input_ids: List[List[int]] = [
        list(base_decoder_ids) for _ in range(batch_size)
    ]
    decoded_params: List[Dict[str, int]] = [{} for _ in range(batch_size)]
    failures: List[Optional[str]] = [None] * batch_size

    for var_name in ordered_names:
        candidate_values_per_b: List[Optional[List[int]]] = [None] * batch_size
        token_masks: List[Optional[torch.Tensor]] = [None] * batch_size
        for b in range(batch_size):
            if failures[b] is not None:
                continue
            try:
                candidate_values = list(oracles[b].candidate_values(var_name))
                if not candidate_values:
                    raise RuntimeError(f"No legal candidates for {var_name}")
                tm = tokenizer.candidate_mask_from_values(
                    var_name, candidate_values, device=device
                )
                if not bool(tm.any()):
                    raise RuntimeError(
                        f"vocab cannot represent any legal candidate for {var_name}"
                    )
                candidate_values_per_b[b] = candidate_values
                token_masks[b] = tm
            except Exception as err:  # pylint: disable=broad-except
                failures[b] = f"{type(err).__name__}: {err}"

        cur_len = len(decoder_input_ids[0])
        step_input = torch.tensor(decoder_input_ids, dtype=torch.long, device=device)
        var_id_row = full_var_ids[:cur_len]
        step_var_ids = torch.tensor(
            [var_id_row] * batch_size, dtype=torch.long, device=device
        )
        logits = model.decode(
            step_input,
            step_var_ids,
            memory,
            zs,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        last_logits = logits[:, -1, :]

        for b in range(batch_size):
            if failures[b] is not None:
                # Keep tensor shapes aligned across the batch.
                decoder_input_ids[b].append(tokenizer.pad_id)
                continue
            try:
                pred_token_id = _sample_token_from_logits(
                    last_logits[b], token_masks[b], options, rng
                )
                pred_value = _resolve_decoded_value(
                    tokenizer, var_name, pred_token_id, candidate_values_per_b[b] or []
                )
                oracles[b].assign(var_name, pred_value)
                decoded_params[b][var_name] = int(pred_value)
                decoder_input_ids[b].append(int(pred_token_id))
            except Exception as err:  # pylint: disable=broad-except
                failures[b] = f"{type(err).__name__}: {err}"
                decoder_input_ids[b].append(tokenizer.pad_id)

    out: List[DecodeRecord] = []
    for b in range(batch_size):
        sym_map = dict(oracles[b].generator.s.sym_map)
        for name, value in decoded_params[b].items():
            sym_map[name] = int(value)
        violations = list(oracles[b].final_violations())
        if failures[b] is not None:
            violations.append(failures[b])
        out.append(
            DecodeRecord(
                params=dict(decoded_params[b]),
                sym_map=sym_map,
                final_violations=violations,
            )
        )
    return out


def make_shifted_zs(
    z0: torch.Tensor,
    direction: torch.Tensor,
    *,
    num_steps: int,
    step_size: float,
    normalize_direction: bool = True,
) -> List[tuple[int, float, torch.Tensor]]:
    direction = direction.detach().to(dtype=torch.float32, device=z0.device)
    if normalize_direction:
        norm = float(direction.norm().item())
        if norm == 0.0:
            raise ValueError("cost vector norm is zero")
        direction = direction / norm

    shifted: List[tuple[int, float, torch.Tensor]] = []
    for step_index in range(num_steps + 1):
        alpha = float(step_index) * float(step_size)
        z = z0 + alpha * direction
        shifted.append((step_index, alpha, z.detach().clone()))
    return shifted



def _convert_scalar_cost_space(
    value: float,
    src: str,
    dst: str,
    task_min_cost: Optional[float],
) -> float:
    """Raw seconds 경유해서 src → dst 공간으로 스칼라 cost 변환."""
    if src == dst:
        return float(value)
    raw = cost_label_to_raw(value, src, task_min_cost=task_min_cost)
    if raw is None:
        return float("nan")
    out = cost_raw_to_label(raw, dst, task_min_cost=task_min_cost)
    return float("nan") if out is None else float(out)


def predict_score(bundle: LoadedBundle, z: torch.Tensor) -> float:
    if bundle.cost_source == "cost_head_gradient":
        z = z.to(device=bundle.device, dtype=torch.float32).view(1, -1)
        return float(bundle.model.cost_head(z).squeeze(-1).item())
    if bundle.cost_weight is None:
        raise RuntimeError("No stored cost vector in checkpoint. Re-run with --latent-gradient.")
    z = z.to(device=bundle.device, dtype=torch.float32)
    raw_pred = float((z @ bundle.cost_weight + bundle.cost_bias).item())
    return _convert_scalar_cost_space(
        raw_pred,
        bundle.ridge_fit_target,
        bundle.ridge_output_target,
        bundle.ridge_task_min_cost,
    )


@dataclass
class CostHeadReEncodePredictor:
    """Uses LatentParamVAE.cost_head(z) directly."""

    model: LatentParamVAE

    def predict(self, z: torch.Tensor) -> float:
        z = z.to(dtype=torch.float32).view(1, -1)
        return float(self.model.cost_head(z).squeeze(-1).item())


@dataclass
class RidgeReEncodePredictor:
    """Linear predictor: ``z @ weight + bias``. The raw output is in
    ``fit_target`` space; ``predict`` returns a value in ``output_target``
    space (defaults to ``fit_target`` if not set). ``task_min_cost`` is
    required whenever a throughput variant appears in either field."""

    weight: torch.Tensor
    bias: float
    name: str = "cost_vec"
    fit_target: str = "neg_log"
    output_target: str = "neg_log"
    task_min_cost: Optional[float] = None

    def predict(self, z: torch.Tensor) -> float:
        z = z.to(device=self.weight.device, dtype=torch.float32).view(-1)
        raw_pred = float((z @ self.weight + self.bias).item())
        return _convert_scalar_cost_space(
            raw_pred, self.fit_target, self.output_target, self.task_min_cost
        )


@torch.no_grad()
def _encode_params_to_z(
    bundle: LoadedBundle,
    ordered_names: List[str],
    params: Dict[str, int],
) -> Optional[torch.Tensor]:
    try:
        ordered_values = [int(params[name]) for name in ordered_names]
    except KeyError:
        return None
    tokenizer = bundle.tokenizer
    enc_ids = torch.tensor(
        [tokenizer.encode_values(ordered_names, ordered_values)],
        dtype=torch.long,
        device=bundle.device,
    )
    enc_var_ids = torch.tensor(
        [tokenizer.encode_var_names(ordered_names)],
        dtype=torch.long,
        device=bundle.device,
    )
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, z_recon, _ = bundle.model.encode(
        enc_ids, enc_var_ids, enc_pad, deterministic=True
    )
    return z_recon[0].to(device=bundle.device, dtype=torch.float32)


@torch.no_grad()
def predict_reencode_score(
    bundle: LoadedBundle,
    ordered_names: List[str],
    params: Dict[str, int],
) -> Optional[float]:
    """Re-encode decoded params and score via the configured re-encode predictor.

    Falls back to cost_head when ``bundle.reencode_predictor`` is None.
    """
    z = _encode_params_to_z(bundle, ordered_names, params)
    if z is None:
        return None
    predictor = bundle.reencode_predictor
    if predictor is None:
        return float(bundle.model.cost_head(z.view(1, -1)).squeeze(-1).item())
    return float(predictor.predict(z))


# Backwards-compat alias — older code paths called this name explicitly.
predict_cost_head_from_reencode = predict_reencode_score


# -----------------------------------------------------------------------------
# Reference-best directory (for printing best-known cost per task)
# -----------------------------------------------------------------------------

_REFERENCE_BEST_CACHE: Dict[str, Dict[str, float]] = {}


def _scan_reference_best_dir(reference_dir: str | Path) -> Dict[str, float]:
    """Scan a directory of measure-record JSONs and return
    ``workload_key → best (lowest) measured mean cost in seconds``.

    Used during latent walk to print a "best known" reference for the task —
    typically pointing to a sibling directory measured on different hardware
    (e.g. ``a6000``) than the training data (``t4``). One scan per process per
    distinct directory thanks to ``_REFERENCE_BEST_CACHE``.
    """
    out: Dict[str, float] = {}
    p = Path(reference_dir)
    if not p.is_dir():
        return out
    no_error = int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    for json_file in sorted(p.glob("*.json")):
        try:
            inputs, results = auto_scheduler.RecordReader(str(json_file)).read_lines()
            inputs = list(inputs)
            results = list(results)
        except Exception:  # pylint: disable=broad-except
            continue
        if not inputs or len(inputs) != len(results):
            continue
        wkey = inputs[0].task.workload_key
        best = float("inf")
        for r in results:
            if int(r.error_no) != no_error:
                continue
            costs = [float(c) for c in r.costs]
            if not costs:
                continue
            mean_s = float(sum(costs) / len(costs))
            if mean_s < best:
                best = mean_s
        if math.isfinite(best):
            prev = out.get(wkey)
            if prev is None or best < prev:
                out[wkey] = best
    return out


def _get_reference_best_seconds(
    reference_dir: Optional[str], workload_key: Optional[str]
) -> Optional[float]:
    """Cached lookup of the best raw-seconds cost for ``workload_key`` in
    ``reference_dir``. Returns ``None`` when the dir is unset / empty / has
    no record matching the workload."""
    if not reference_dir or not workload_key:
        return None
    cache_key = str(Path(reference_dir).resolve())
    cached = _REFERENCE_BEST_CACHE.get(cache_key)
    if cached is None:
        cached = _scan_reference_best_dir(reference_dir)
        _REFERENCE_BEST_CACHE[cache_key] = cached
    return cached.get(workload_key)


@torch.no_grad()
def predict_recon_score(
    bundle: LoadedBundle,
    ordered_names: List[str],
    params: Dict[str, int],
) -> tuple[Optional[float], Optional[float]]:
    """Re-encode decoded params through the encoder and score the resulting z.

    Grounds predictions to the encoder's training manifold so cost_head/ridge
    evaluates an in-distribution latent instead of the OOD walked z.

    Returns ``(mean, std)``. ``std`` is the GP posterior standard deviation
    when ``bundle.recon_predictor`` is the GP variant; otherwise ``None``.
    """
    try:
        ordered_values = [int(params[name]) for name in ordered_names]
    except KeyError:
        return None, None
    tokenizer = bundle.tokenizer
    enc_ids = torch.tensor(
        [tokenizer.encode_values(ordered_names, ordered_values)],
        dtype=torch.long,
        device=bundle.device,
    )
    enc_var_ids = torch.tensor(
        [tokenizer.encode_var_names(ordered_names)],
        dtype=torch.long,
        device=bundle.device,
    )
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    _, _, z_recon, _ = bundle.model.encode(
        enc_ids, enc_var_ids, enc_pad, deterministic=True
    )
    z = z_recon[0].to(device=bundle.device, dtype=torch.float32).view(1, -1)
    if bundle.recon_predictor is not None:
        if hasattr(bundle.recon_predictor, "predict_with_std"):
            mean, std = bundle.recon_predictor.predict_with_std(z[0])
            return float(mean), float(std)
        return float(bundle.recon_predictor.predict(z[0])), None
    return float(bundle.model.cost_head(z).squeeze(-1).item()), None


def compute_walk_direction(bundle: LoadedBundle, z0: torch.Tensor) -> torch.Tensor:
    if bundle.cost_source == "missing_cost_vector":
        raise RuntimeError("No stored cost vector in checkpoint. Re-run with --latent-gradient.")
    if bundle.cost_weight is not None:
        return bundle.cost_weight.detach().to(dtype=torch.float32, device=z0.device)

    z = z0.detach().to(device=bundle.device, dtype=torch.float32).view(1, -1).requires_grad_(True)
    cost_pred = bundle.model.cost_head(z).sum()
    grad = torch.autograd.grad(cost_pred, z, retain_graph=False, create_graph=False)[0]
    return grad[0].detach().to(device=z0.device, dtype=torch.float32)



def build_walk_records(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
    *,
    num_steps: int,
    step_size: float,
    normalize_direction: bool = True,
    random_z: bool = False,
    seed: Optional[int] = None,
    output: Optional[str | Path] = None,
    deterministic_start: bool = False,
    include_recon_predict: bool = False,
    include_measurement: bool = True,
    keep_bundle: bool = False,
    measurement_cache: Optional[Dict[tuple, Any]] = None,
    sampling_options: Optional[SamplingOptions] = None,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
    sort_by: str = "re_pred",
    show_neg_log: bool = False,
    reference_best_seconds: Optional[float] = None,
    reference_best_dir: Optional[str] = None,
) -> tuple[List[WalkRecord], StartZContext]:
    options = sampling_options or SamplingOptions()
    decode_rng: Optional[torch.Generator] = None
    if options.strategy != "greedy" and options.seed is not None:
        decode_rng = torch.Generator(device=bundle.device)
        decode_rng.manual_seed(int(options.seed))
    start_ctx = resolve_start_z(
        bundle,
        record,
        random_z=random_z,
        seed=seed,
        deterministic_start=deterministic_start,
    )
    z0 = start_ctx.z0
    gen = start_ctx.generator
    ordered_names = start_ctx.ordered_names
    task_for_measure = gen._task
    measure_record_path = _resolve_walk_measure_record_path(
        record=record,
        task=task_for_measure,
        output=output,
    )
    walk_direction = compute_walk_direction(bundle, z0)
    shifted_zs = make_shifted_zs(
        z0,
        walk_direction,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
    )

    ref_params = {
        str(k): int(v) for k, v in (record.params or {}).items() if isinstance(v, int)
    }

    def _is_reference_sym_map(sym_map: Dict[str, int]) -> bool:
        if not ref_params or not sym_map:
            return False
        shared = set(ref_params) & set(sym_map)
        if not shared:
            return False
        return all(int(sym_map[k]) == ref_params[k] for k in shared)

    results: List[WalkRecord] = []
    pending_measure_indices_by_key: Dict[Any, List[int]] = {}
    pending_measure_states: List[Any] = []
    pending_measure_metas: List[Dict[str, Any]] = []
    pending_measure_keys: List[Any] = []
    shape_token_ids, shape_var_ids = shape_prefix_for_record(bundle, record)
    extent_token_ids, extent_var_ids = extent_prefix_for_record(bundle, record, gen=gen)

    # Decode all walk steps in a single batched forward pass. The transformer
    # work scales O(num_steps × num_params) per call; batching turns that
    # into one call per param position with batch size = num_steps, which is
    # the dominant speedup.
    step_oracles = [
        bundle.registry.build_oracle_from_record(record) for _ in shifted_zs
    ]
    zs_stack = torch.stack(
        [z for _, _, z in shifted_zs], dim=0
    ).to(device=bundle.device, dtype=torch.float32).reshape(len(shifted_zs), -1)
    decoded_per_step = greedy_decode_from_zs_batch(
        bundle,
        step_oracles,
        ordered_names,
        zs_stack,
        sampling_options=options,
        rng=decode_rng,
        shape_token_ids=shape_token_ids,
        shape_var_ids=shape_var_ids,
        extent_token_ids=extent_token_ids,
        extent_var_ids=extent_var_ids,
    )

    for (step_index, alpha, z), decoded, oracle in zip(
        shifted_zs, decoded_per_step, step_oracles
    ):

        decoded_sym_map_int = {
            k: int(v) for k, v in decoded.sym_map.items() if isinstance(v, int)
        }
        # Task-aware cache key prevents cross-workload contamination of the
        # measurement cache. Two distinct workloads frequently produce
        # overlapping sym_maps and a sym_map-only key would let a small task's
        # fast cost leak into a large task's walk via the cache.
        sym_key = make_task_sym_map_key(
            getattr(record, "workload_key", None), decoded_sym_map_int
        )
        is_reference_sym = _is_reference_sym_map(decoded_sym_map_int)

        state = None
        state_build_ok = False
        state_build_error = None
        measurement = None
        predicted_score = predict_score(bundle, z)

        if is_reference_sym:
            measurement = {
                "ok": True,
                "usable_measurement": True,
                "mean_cost": 0.0,
                "reference_skip": True,
            }
        elif not decoded.final_violations:
            try:
                state = gen.params_to_state(decoded.params)
                state_build_ok = True
            except Exception as err:  # pylint: disable=broad-except
                state_build_error = f"{type(err).__name__}: {err}"

        measure_meta = {
            "sample_id": record.sample_id,
            "json_path": record.json_path,
            "task_index": record.task_index,
            "workload_key": record.workload_key,
            "target_kind": record.target_kind,
            "sketch_index": record.sketch_index,
            "step_index": step_index,
            "alpha": alpha,
            "predicted_score": predicted_score,
        }
        if state_build_ok:
            cached_measurement = (
                measurement_cache.get(sym_key) if measurement_cache is not None else None
            )
            if cached_measurement is not None:
                measurement = dict(cached_measurement)
            elif sym_key in pending_measure_indices_by_key:
                pending_measure_indices_by_key[sym_key].append(len(results))
            else:
                pending_measure_indices_by_key[sym_key] = [len(results)]
                pending_measure_keys.append(sym_key)
                pending_measure_states.append(state)
                pending_measure_metas.append(measure_meta)

        recon_predict_cost: Optional[float] = None
        recon_predict_std: Optional[float] = None
        if include_recon_predict and not decoded.final_violations:
            try:
                recon_predict_cost, recon_predict_std = predict_recon_score(
                    bundle, ordered_names, decoded.params
                )
            except Exception as err:  # pylint: disable=broad-except
                print(f"[latent-walk] recon predict failed at step={step_index}: {type(err).__name__}: {err}")
                recon_predict_cost = None
                recon_predict_std = None

        reencode_cost_pred: Optional[float] = None
        if not decoded.final_violations:
            try:
                reencode_cost_pred = predict_reencode_score(
                    bundle, ordered_names, decoded.params
                )
            except Exception as err:  # pylint: disable=broad-except
                print(
                    f"[latent-walk] reencode cost_head failed at step={step_index}: "
                    f"{type(err).__name__}: {err}"
                )
                reencode_cost_pred = None

        walk_record = WalkRecord(
            step_index=step_index,
            alpha=alpha,
            predicted_score=predicted_score,
            z=z.detach().cpu().tolist(),
            params=dict(decoded.params),
            sym_map={k: int(v) for k, v in decoded.sym_map.items() if isinstance(v, int)},
            final_violations=list(decoded.final_violations),
            state_build_ok=state_build_ok,
            state_build_error=state_build_error,
            measurement=measurement,
            recon_predict_cost=recon_predict_cost,
            recon_predict_std=recon_predict_std,
            reencode_cost_pred=reencode_cost_pred,
        )
        results.append(walk_record)

    unique_sym_count = len({
        frozenset(rec.sym_map.items())
        for rec in results
        if not _is_reference_sym_map(rec.sym_map)
    })
    ref_match_count = sum(
        1 for rec in results if _is_reference_sym_map(rec.sym_map)
    )
    print(
        f"Walk records: unique sym_map={unique_sym_count} "
        f"total={len(results)} "
        f"ref_skipped={ref_match_count} "
    )
    # ``record.cost`` lives in the loader's training-label space (default
    # ``neg_log``); convert back to raw seconds for human-readable output.
    ref_raw_cost = cost_label_to_raw(record.cost, "neg_log")
    ref_cost_str = (
        f"{float(ref_raw_cost):.6g}s" if ref_raw_cost is not None else "<none>"
    )
    print(f"Reference cost: {ref_cost_str} (sample_id={record.sample_id})")
    if reference_best_seconds is not None:
        ref_label = (
            Path(reference_best_dir).name
            if reference_best_dir
            else "reference"
        )
        print(f"{ref_label} measured best: {float(reference_best_seconds):.6g}s")

    _release_measurement_environment(bundle, generator=gen, keep_bundle=keep_bundle)

    if include_measurement:
        # Hand the GPU over to the measurement subprocesses. When the bundle is
        # being reused across walks (keep_bundle=True), the model is still
        # resident on the original device — temporarily evict it so measurement
        # gets the full VRAM, then restore afterwards.
        evicted_model_device: Optional[torch.device] = None
        if keep_bundle and bundle.model is not None and bundle.device.type == "cuda":
            evicted_model_device = bundle.device
            bundle.model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        try:
            measurement_session = _build_measurement_session(
                task_for_measure, output_path=measure_record_path
            )
            measured_results = measure_candidates_batch(
                session=measurement_session,
                task=task_for_measure,
                states=pending_measure_states,
                metas=pending_measure_metas,
                cost_target=cost_target,
                task_min_cost=task_min_cost,
            )
            for sym_key, measurement in zip(pending_measure_keys, measured_results):
                for result_index in pending_measure_indices_by_key[sym_key]:
                    results[result_index].measurement = dict(measurement)
                if measurement_cache is not None:
                    prior = measurement_cache.get(sym_key)
                    if prior is None:
                        measurement_cache[sym_key] = dict(measurement)
                    else:
                        prior_ok = bool(prior.get("ok") and prior.get("usable_measurement"))
                        curr_ok = bool(measurement.get("ok") and measurement.get("usable_measurement"))
                        prior_mc = prior.get("mean_cost") if prior_ok else None
                        curr_mc = measurement.get("mean_cost") if curr_ok else None
                        if curr_mc is not None and (prior_mc is None or curr_mc > prior_mc):
                            measurement_cache[sym_key] = dict(measurement)

            print("[latent-walk] measurement results")
            grouped = _group_walk_records_by_sym_map(results)
            if sort_by == "measured":
                # Ascending raw seconds: fastest measured candidates first.
                # Records without a usable measurement fall to the end.
                def _measured_key(g):
                    meas = (g["record"].measurement or {})
                    raw_costs = meas.get("costs") or []
                    if not raw_costs or not meas.get("usable_measurement"):
                        return float("inf")
                    return float(sum(float(c) for c in raw_costs) / len(raw_costs))
                grouped.sort(key=_measured_key)
            elif sort_by == "alpha":
                # Ascending walk progression: anchor (alpha=0) first, then
                # outward. Within a group we use the smallest alpha at which
                # that sym_map first appeared.
                grouped.sort(
                    key=lambda g: min(g["alphas"]) if g["alphas"] else float("inf")
                )
            else:  # "re_pred"
                grouped.sort(
                    key=lambda g: (
                        g["record"].reencode_cost_pred
                        if g["record"].reencode_cost_pred is not None
                        else float("-inf")
                    ),
                    reverse=True,
                )
            reencode_label = f"re_pred[{bundle.reencode_predictor_name}]"
            for grouped_record in grouped:
                log_grouped_candidate_result(
                    grouped_record,
                    reencode_label=reencode_label,
                    show_neg_log=show_neg_log,
                )
            print("=" * 80)
        finally:
            if evicted_model_device is not None:
                bundle.model.to(evicted_model_device)

    return results, start_ctx



def _group_walk_records_by_sym_map(records: List[WalkRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[tuple[str, int], ...], Dict[str, Any]] = {}
    order: List[tuple[tuple[str, int], ...]] = []
    for record in records:
        key = make_sym_map_key(record.sym_map)
        group = grouped.get(key)
        if group is None:
            group = {
                "record": record,
                "alphas": [],
                "pred_costs": [],
            }
            grouped[key] = group
            order.append(key)
        group["alphas"].append(record.alpha)
        group["pred_costs"].append(record.predicted_score)
    return [grouped[key] for key in order]


LATENT_RESULT_FIELDNAMES = [
    "sample_id",
    "best_cost",
    "random",
    "seed",
    "encode_deterministic",
    "encoded_z",
    "alpha",
    "pred_cost",
    "mean_cost",
    "true_mean_cost",
    "latent_gradient",
]


def _append_latent_result_csv(
    *,
    csv_path: str | Path,
    record: JsonSampleRecord,
    records: List[WalkRecord],
    start_ctx: StartZContext,
    best_cost: bool,
    random_z: bool,
    seed: Optional[int],
    latent_gradient: bool,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
) -> None:
    if random_z and seed is not None:
        existing_random_seeds = load_existing_random_seeds(csv_path)
        if int(seed) in existing_random_seeds:
            print(
                "[latent-walk] skip csv append because random record with same seed already exists "
                f"seed={seed}"
            )
            return

    if start_ctx.encode_deterministic is True and best_cost:
        existing_deterministic_samples = load_existing_deterministic_sample_ids(csv_path)
        if record.sample_id in existing_deterministic_samples:
            print(
                "[latent-walk] skip csv append because deterministic record already exists "
                f"sample_id={record.sample_id}"
            )
            return

    true_mean_cost = (
        None
        if random_z
        else extract_true_mean_cost(
            record,
            cost_target=cost_target,
            task_min_cost=task_min_cost,
        )
    )
    rows: List[Dict[str, Any]] = []
    for grouped_record in _group_walk_records_by_sym_map(records):
        walk_record = grouped_record["record"]
        measurement = walk_record.measurement or {}
        usable_measurement = bool(
            measurement.get("ok") and measurement.get("usable_measurement")
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "best_cost": best_cost,
                "random": random_z,
                "seed": seed,
                "encode_deterministic": start_ctx.encode_deterministic,
                "encoded_z": (
                    start_ctx.encoded_z
                    if start_ctx.encode_deterministic is False
                    else None
                ),
                "alpha": grouped_record["alphas"],
                "pred_cost": grouped_record["pred_costs"],
                "mean_cost": measurement.get("mean_cost") if usable_measurement else None,
                "true_mean_cost": (
                    true_mean_cost
                    if (not random_z and 0.0 in grouped_record["alphas"])
                    else None
                ),
                "latent_gradient": latent_gradient,
            }
        )

    append_result_rows(csv_path, LATENT_RESULT_FIELDNAMES, rows)



def run_latent_walk(
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    *,
    network_info_folder: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 8,
    step_size: float = 0.25,
    normalize_direction: bool = True,
    random_z: bool = False,
    seed: Optional[int] = None,
    best_cost: bool = False,
    output: Optional[str | Path] = None,
    latent_gradient: bool = False,
    deterministic_start: bool = False,
    preselected_record: Optional[JsonSampleRecord] = None,
    include_recon_predict: bool = False,
    include_measurement: bool = True,
    bundle: Optional[LoadedBundle] = None,
    keep_bundle: bool = False,
    measurement_cache: Optional[Dict[tuple, Any]] = None,
    sampling_options: Optional[SamplingOptions] = None,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
    sort_by: str = "re_pred",
    show_neg_log: bool = False,
    reference_best_dir: Optional[str] = None,
) -> List[WalkRecord]:
    if bundle is None:
        bundle = load_bundle(
            checkpoint_path,
            network_info_folder=network_info_folder,
            device=device,
            use_latent_gradient=latent_gradient,
        )
    if bundle.cost_source == "missing_cost_vector":
        print("[latent-walk] checkpoint에 저장된 cost vector가 없습니다.")
        return []
    if preselected_record is not None:
        record = preselected_record
    else:
        record = _select_record_from_path(record_json_path, best_cost=best_cost)
    reference_best_seconds = _get_reference_best_seconds(
        reference_best_dir, getattr(record, "workload_key", None)
    )
    records, start_ctx = build_walk_records(
        bundle,
        record,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
        random_z=random_z,
        seed=seed,
        output=output,
        deterministic_start=deterministic_start,
        include_recon_predict=include_recon_predict,
        include_measurement=include_measurement,
        keep_bundle=keep_bundle,
        measurement_cache=measurement_cache,
        sampling_options=sampling_options,
        cost_target=cost_target,
        task_min_cost=task_min_cost,
        sort_by=sort_by,
        show_neg_log=show_neg_log,
        reference_best_seconds=reference_best_seconds,
        reference_best_dir=reference_best_dir,
    )
    csv_output_path = resolve_results_csv_path(
        __file__,
        "by_latent",
        checkpoint_path,
        bundle.checkpoint_payload,
    )
    _append_latent_result_csv(
        csv_path=csv_output_path,
        record=record,
        records=records,
        start_ctx=start_ctx,
        best_cost=best_cost,
        random_z=random_z,
        seed=seed,
        latent_gradient=latent_gradient,
        cost_target=cost_target,
        task_min_cost=task_min_cost,
    )
    return records

