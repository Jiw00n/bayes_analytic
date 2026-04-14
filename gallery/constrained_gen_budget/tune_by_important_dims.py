from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tvm import auto_scheduler

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))

    try:
        from latent_model_budget.adapter import (
            GeneratorRegistry,
            JsonSampleRecord,
            LegalPrefixOracle,
            load_json_sample,
            load_json_samples,
            split_records,
        )
        from latent_model_budget.config import build_config
        from latent_model_budget.dataset import budget_enabled, get_model_param_order
        from latent_model_budget.model import LatentParamVAE
        from latent_model_budget.tokenizer import ParamTokenizer
    except ImportError:
        from latent_model_budget.adapter import (
            GeneratorRegistry,
            JsonSampleRecord,
            LegalPrefixOracle,
            load_json_sample,
            load_json_samples,
            split_records,
        )
        from latent_model_budget.config import build_config
        from latent_model_budget.dataset import budget_enabled, get_model_param_order
        from latent_model_budget.model import LatentParamVAE
        from latent_model_budget.tokenizer import ParamTokenizer

    from modules.task_paths import clean_name, get_measure_record_filename
    from result_csv_utils import (
        append_result_rows,
        extract_true_mean_cost,
        load_existing_random_seeds,
        make_sym_map_key,
        resolve_results_csv_path,
    )
else:
    try:
        from .latent_model_budget.adapter import (
            GeneratorRegistry,
            JsonSampleRecord,
            LegalPrefixOracle,
            load_json_sample,
            load_json_samples,
            split_records,
        )
        from .latent_model_budget.config import build_config
        from .latent_model_budget.dataset import budget_enabled, get_model_param_order
        from .latent_model_budget.model import LatentParamVAE
        from .latent_model_budget.tokenizer import ParamTokenizer
    except ImportError:
        from .latent_model_budget.adapter import (
            GeneratorRegistry,
            JsonSampleRecord,
            LegalPrefixOracle,
            load_json_sample,
            load_json_samples,
            split_records,
        )
        from .latent_model_budget.config import build_config
        from .latent_model_budget.dataset import budget_enabled, get_model_param_order
        from .latent_model_budget.model import LatentParamVAE
        from .latent_model_budget.tokenizer import ParamTokenizer

    from .modules.task_paths import clean_name, get_measure_record_filename
    from .result_csv_utils import (
        append_result_rows,
        extract_true_mean_cost,
        load_existing_random_seeds,
        make_sym_map_key,
        resolve_results_csv_path,
    )


# -----------------------------------------------------------------------------
# Defaults from intervention diagnosis
# -----------------------------------------------------------------------------

# DEFAULT_IMPORTANT_DIMS: List[int] = [56, 1, 15, 17, 19, 43, 45, 42, 33, 39]
# DEFAULT_IMPORTANT_DIMS: List[int] = [1, 4, 17, 34, 38, 39, 40, 45, 12, 15, 24, 48, 61]
DEFAULT_IMPORTANT_DIMS: List[int] = [43, 42, 15, 39, 17, 40, 38]
DEFAULT_ALPHAS: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
DEFAULT_ALPHAS: List[float] = [i * 0.25 for i in range(1, 21)]

DEFAULT_CHECKPOINT_PATH = "/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/last.pt"


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


@dataclass
class DimInterventionRecord:
    candidate_index: int
    alpha: float
    coefficient_source: str
    coefficient_by_dim: Dict[int, float]
    predicted_score: float
    z: List[float]
    params: Dict[str, int]
    sym_map: Dict[str, int]
    final_violations: List[str]
    state_build_ok: bool
    state_build_error: Optional[str]
    measurement: Any


def _task_repeat(task: Any) -> int:
    return 3


def _make_measurer(task: Any, log_filename: str) -> Any:
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


def _build_measurement_session(task: Any, measure_output_dir: Optional[str] = None) -> MeasurementSession:
    output_path = str(get_measure_record_filename(task, task.target, output_dir=measure_output_dir))
    return MeasurementSession(
        task=task,
        output_path=output_path,
        measurer=_make_measurer(task, output_path),
        policy=auto_scheduler.search_policy.EmptyPolicy(task),
    )


def measure_candidate(*, session: MeasurementSession, result: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    error_no = int(result.error_no)
    costs = [float(x) for x in result.costs]
    mean_cost = -math.log(float(sum(costs) / len(costs))) if costs else None
    usable_measurement = error_no == int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    error_msg = str(result.error_msg)
    return {
        "ok": True,
        "usable_measurement": usable_measurement,
        "error_no": error_no,
        "error_msg": error_msg or None,
        "costs": costs,
        "mean_cost": mean_cost,
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
) -> List[Dict[str, Any]]:
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
        measure_candidate(result=result, meta=meta, session=session)
        for result, meta in zip(results, metas)
    ]


def _default_output_path(
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    output_dir: str | Path,
) -> Path:
    checkpoint_stem = clean_name(Path(checkpoint_path).stem)
    record_stem = clean_name(Path(record_json_path).stem)
    return Path(output_dir) / "important_dim_records" / f"{checkpoint_stem}__{record_stem}.jsonl"


def _resolve_output_layout(
    *,
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    output: Optional[str | Path],
) -> tuple[Optional[Path], Optional[str]]:
    if output is None:
        return None, None
    output_root = Path(output)
    record_output_path = _default_output_path(checkpoint_path, record_json_path, output_root)
    measure_output_dir = str(output_root / "measure_records")
    return record_output_path, measure_output_dir


def _resolve_measure_output_dir(output: Optional[str | Path]) -> Optional[str]:
    if output is None:
        return None
    return str(Path(output) / "measure_records")


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
) -> None:
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

    bundle.model = None
    bundle.tokenizer = None
    bundle.cost_weight = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -----------------------------------------------------------------------------
# Model / decode helpers
# -----------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _make_model_cfg(cfg_payload: Dict[str, Any]) -> Any:
    default_model_cfg = build_config().model
    merged = {
        key: value
        for key, value in vars(default_model_cfg).items()
    }
    merged.update(dict(cfg_payload))
    return SimpleNamespace(**merged)


def load_bundle(
    checkpoint_path: str | Path,
    *,
    network_info_folder: Optional[str] = None,
    device: str = "cuda",
) -> LoadedBundle:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = ParamTokenizer.from_checkpoint_payload(payload)
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
    cost_source = "cost_head"
    if latent_cost_ridge is not None and "weight" in latent_cost_ridge:
        cost_weight = latent_cost_ridge["weight"].detach().to(dtype=torch.float32, device=torch_device)
        cost_bias = float(latent_cost_ridge.get("bias", 0.0))
        cost_source = "latent_cost_ridge"

    if network_info_folder is None:
        network_info_folder = payload["config"]["data"]["network_info_folder"]

    registry = GeneratorRegistry(network_info_folder)
    return LoadedBundle(
        checkpoint_payload=payload,
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        cost_weight=cost_weight,
        cost_bias=cost_bias,
        cost_source=cost_source,
        device=torch_device,
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
def encode_record_to_z(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, Any, List[str], List[int]]:
    gen, ordered_names, ordered_values = prepare_record_context(bundle, record)

    enc_ids = torch.tensor(
        [bundle.tokenizer.encode_values(ordered_names, ordered_values)],
        dtype=torch.long,
        device=bundle.device,
    )
    enc_var_ids = torch.tensor(
        [bundle.tokenizer.encode_var_names(ordered_names)],
        dtype=torch.long,
        device=bundle.device,
    )
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
def greedy_decode_from_z(
    bundle: LoadedBundle,
    oracle: LegalPrefixOracle,
    ordered_names: List[str],
    z: torch.Tensor,
) -> DecodeRecord:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    z = z.to(device=device, dtype=torch.float32).view(1, -1)
    memory = model.latent_to_memory(z).view(1, model.cfg.latent_token_count, model.cfg.d_model)

    decoder_input_ids: List[int] = [tokenizer.bos_id]
    decoded_params: Dict[str, int] = {}

    for var_name in ordered_names:
        candidate_values = list(oracle.candidate_values(var_name))
        if not candidate_values:
            raise RuntimeError(f"No legal candidates returned for {var_name}")
        if len(candidate_values) == 1:
            pred_value = int(candidate_values[0])
            pred_token_id = int(
                tokenizer.token_to_id.get(
                    tokenizer.value_to_token(var_name, pred_value),
                    tokenizer.unk_id,
                )
            )
            oracle.assign(var_name, pred_value)
            decoded_params[var_name] = int(pred_value)
            decoder_input_ids.append(pred_token_id)
            continue

        token_mask = tokenizer.candidate_mask_from_values(var_name, candidate_values, device=device)
        if not bool(token_mask.any()):
            raise RuntimeError(
                f"Token vocab cannot represent any legal candidate for {var_name}. "
                f"candidates={candidate_values}"
            )

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
            z,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[0, -1].masked_fill(~token_mask, float("-inf"))
        pred_token_id = int(torch.argmax(step_logits).item())
        pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

        oracle.assign(var_name, pred_value)
        decoded_params[var_name] = int(pred_value)
        decoder_input_ids.append(pred_token_id)

    sym_map = dict(oracle.generator.s.sym_map)
    for name, value in decoded_params.items():
        sym_map[name] = int(value)

    return DecodeRecord(
        params=decoded_params,
        sym_map=sym_map,
        final_violations=list(oracle.final_violations()),
    )


@torch.no_grad()
def predict_score(bundle: LoadedBundle, z: torch.Tensor) -> float:
    if bundle.cost_weight is not None:
        z = z.to(device=bundle.device, dtype=torch.float32)
        return float((z @ bundle.cost_weight + bundle.cost_bias).item())
    z = z.to(device=bundle.device, dtype=torch.float32).view(1, -1)
    return float(bundle.model.cost_head(z).squeeze(-1).item())


# -----------------------------------------------------------------------------
# Important-dim intervention helpers
# -----------------------------------------------------------------------------


def _expand_json_paths(entries: Sequence[str]) -> List[Path]:
    raw_paths: List[Path] = []
    for entry in entries:
        p = Path(entry)
        if p.is_dir():
            raw_paths.extend(sorted(p.glob("*.json")))
        else:
            raw_paths.append(p)
    return raw_paths


def _load_train_records_for_coefficients(bundle: LoadedBundle) -> List[JsonSampleRecord]:
    cfg_data = dict(bundle.checkpoint_payload.get("config", {}).get("data", {}))
    json_paths = list(cfg_data.get("json_paths", []))
    if not json_paths:
        return []

    records: List[JsonSampleRecord] = []
    for path in _expand_json_paths(json_paths):
        records.extend(load_json_samples(path))
    if not records:
        return []

    if bool(cfg_data.get("filter_invalid_records", False)):
        filtered: List[JsonSampleRecord] = []
        include_budget = budget_enabled({"data": cfg_data})
        for record in records:
            oracle = bundle.registry.build_oracle_from_record(record)
            gen = oracle.generator
            order = get_model_param_order(gen, include_budget=include_budget)
            values = [int(record.params[name]) for name in order]
            if oracle.validate_assignment(order, values):
                filtered.append(record)
        records = filtered

    train_records, _, _ = split_records(
        records,
        float(cfg_data.get("train_ratio", 0.9)),
        float(cfg_data.get("val_ratio", 0.1)),
        float(cfg_data.get("test_ratio", 0.0)),
        int(cfg_data.get("seed", 42)),
    )
    return list(train_records)


def _fit_important_dim_coefficients_from_train(
    bundle: LoadedBundle,
    important_dims: Sequence[int],
) -> tuple[Dict[int, float], str]:
    dims = [int(dim) for dim in important_dims]
    if not dims:
        raise ValueError("important_dims must not be empty")

    if bundle.cost_weight is not None:
        coeffs = {dim: float(bundle.cost_weight[int(dim)].item()) for dim in dims}
        return coeffs, "checkpoint_latent_cost_ridge"

    train_records = _load_train_records_for_coefficients(bundle)
    if not train_records:
        raise RuntimeError("Could not load train records to fit important-dim coefficients")

    latent_rows: List[torch.Tensor] = []
    cost_rows: List[float] = []
    for record in train_records:
        if record.cost is None:
            continue
        z, _, _, _ = encode_record_to_z(bundle, record)
        latent_rows.append(z[dims].detach().cpu().to(dtype=torch.float64))
        cost_rows.append(float(record.cost))

    if not latent_rows:
        raise RuntimeError("No finite train costs available to fit important-dim coefficients")

    x = torch.stack(latent_rows, dim=0)
    y = torch.tensor(cost_rows, dtype=torch.float64)
    ones = torch.ones((int(x.shape[0]), 1), dtype=torch.float64)
    design = torch.cat([x, ones], dim=1)

    ridge_alpha_raw = (
        bundle.checkpoint_payload.get("config", {})
        .get("train", {})
        .get("ridge_alpha", 0.1)
    )
    if isinstance(ridge_alpha_raw, (list, tuple)):
        ridge_alpha = float(ridge_alpha_raw[0]) if ridge_alpha_raw else 0.1
    else:
        ridge_alpha = float(ridge_alpha_raw)
    reg = torch.eye(int(design.shape[1]), dtype=torch.float64) * ridge_alpha
    reg[-1, -1] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ y
    try:
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.pinv(lhs) @ rhs

    weight = coeff[:-1]
    coeffs = {dim: float(weight[idx].item()) for idx, dim in enumerate(dims)}
    return coeffs, "train_ridge_fit"


def _unit_important_dim_coefficients(
    important_dims: Sequence[int],
) -> tuple[Dict[int, float], str]:
    dims = [int(dim) for dim in important_dims]
    if not dims:
        raise ValueError("important_dims must not be empty")
    return {dim: 1.0 for dim in dims}, "unit_coefficients"


def _normalize_coefficients_l2(
    coefficient_by_dim: Dict[int, float],
) -> Dict[int, float]:
    if not coefficient_by_dim:
        raise ValueError("coefficient_by_dim must not be empty")
    norm_sq = 0.0
    for value in coefficient_by_dim.values():
        norm_sq += float(value) * float(value)
    norm = norm_sq ** 0.5
    if norm <= 0.0:
        raise ValueError("coefficient vector has zero L2 norm")
    return {
        int(dim): float(value) / norm
        for dim, value in coefficient_by_dim.items()
    }


def make_dim_intervention_zs(
    z0: torch.Tensor,
    *,
    coefficient_by_dim: Dict[int, float],
    alphas: List[float],
) -> List[tuple[int, float, torch.Tensor]]:
    shifted: List[tuple[int, float, torch.Tensor]] = []
    shifted.append((0, 0.0, z0.detach().clone()))

    candidate_index = 1
    for alpha in alphas:
        z = z0.detach().clone()
        for dim, coeff in coefficient_by_dim.items():
            z[int(dim)] = z[int(dim)] + float(alpha) * float(coeff)
        shifted.append((candidate_index, float(alpha), z))
        candidate_index += 1
    return shifted


def _log_intervention_result(result: Dict[str, Any]) -> None:
    candidate_index = result["candidate_index"]
    alpha = result["alpha"]
    predicted_score = result["predicted_score"]
    coefficient_source = result["coefficient_source"]
    coefficient_by_dim = result.get("coefficient_by_dim") or {}

    tag = "base" if candidate_index == 0 else f"combined dims={len(coefficient_by_dim)}"
    prefix = (
        f"[important-dim] idx={candidate_index} {tag} "
        f"alpha={alpha:.4f} pred={predicted_score:.6f} "
        f"coeff_source={coefficient_source}"
    )

    if result["final_violations"]:
        print(f"{prefix} status=violated violations={len(result['final_violations'])}")
        return

    if not result["state_build_ok"]:
        print(f"{prefix} status=state_build_failed error={result['state_build_error']}")
        return

    measurement = result.get("measurement") or {}
    if not measurement.get("ok"):
        print(f"{prefix} status=measure_failed error={measurement.get('error')}")
        return

    if not measurement.get("usable_measurement"):
        print(
            f"{prefix} status=measure_error "
            f"error_no={measurement.get('error_no')} "
            f"error_msg={measurement.get('error_msg')}"
        )
        return

    mean_cost = measurement.get("mean_cost")
    mean_cost_text = "n/a" if mean_cost is None else f"{mean_cost:.9f}"
    print(
        f"{prefix} status=ok mean_cost={mean_cost_text} "
        f"log={measurement.get('measure_record_path')}"
    )


def _log_grouped_intervention_result(grouped_record: Dict[str, Any]) -> None:
    record = grouped_record["record"]
    alphas = grouped_record["alphas"]
    pred_costs = grouped_record["pred_costs"]
    coefficient_source = record.coefficient_source
    alpha_text = json.dumps(alphas, ensure_ascii=False)
    pred_mean = (
        float(sum(float(x) for x in pred_costs) / len(pred_costs))
        if pred_costs
        else None
    )
    pred_mean_text = "n/a" if pred_mean is None else f"{pred_mean:.6f}"

    prefix = (
        f"[important-dim] alphas={alpha_text} "
        f"pred_mean={pred_mean_text} "
        f"coeff_source={coefficient_source}"
    )

    if record.final_violations:
        print(f"{prefix} status=violated violations={len(record.final_violations)}")
        return

    if not record.state_build_ok:
        print(f"{prefix} status=state_build_failed error={record.state_build_error}")
        return

    measurement = record.measurement or {}
    if not measurement.get("ok"):
        print(f"{prefix} status=measure_failed error={measurement.get('error')}")
        return

    if not measurement.get("usable_measurement"):
        print(
            f"{prefix} status=measure_error "
            f"error_no={measurement.get('error_no')} "
            f"error_msg={measurement.get('error_msg')}"
        )
        return

    mean_cost = measurement.get("mean_cost")
    mean_cost_text = "n/a" if mean_cost is None else f"{mean_cost:.9f}"
    print(
        f"{prefix} status=ok mean_cost={mean_cost_text} "
        f"log={measurement.get('measure_record_path')}"
    )


def build_important_dim_records(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
    *,
    important_dims: Sequence[int],
    alphas: List[float],
    use_unit_coefficients: bool = False,
    random_z: bool = False,
    seed: Optional[int] = None,
    output: Optional[str | Path] = None,
    deterministic_start: bool = False,
) -> tuple[List[DimInterventionRecord], StartZContext]:
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
    measure_output_dir = _resolve_measure_output_dir(output)
    task_for_measure = gen._task
    if use_unit_coefficients:
        coefficient_by_dim, coefficient_source = _unit_important_dim_coefficients(important_dims)
    else:
        coefficient_by_dim, coefficient_source = _fit_important_dim_coefficients_from_train(
            bundle,
            important_dims,
        )
    coefficient_by_dim = _normalize_coefficients_l2(coefficient_by_dim)
    coefficient_source = f"{coefficient_source}_l2_normalized"
    print(
        "[important-dim] learned_coefficients "
        f"source={coefficient_source} "
        f"values={{{', '.join(f'{int(dim)}:{float(coeff):.6f}' for dim, coeff in sorted(coefficient_by_dim.items()))}}}"
    )
    shifted_zs = make_dim_intervention_zs(
        z0,
        coefficient_by_dim=coefficient_by_dim,
        alphas=alphas,
    )

    results: List[DimInterventionRecord] = []
    pending_measure_indices_by_key: Dict[tuple[tuple[str, int], ...], List[int]] = {}
    pending_measure_states: List[Any] = []
    pending_measure_metas: List[Dict[str, Any]] = []
    pending_measure_keys: List[tuple[tuple[str, int], ...]] = []

    for candidate_index, alpha, z in shifted_zs:
        oracle = bundle.registry.build_oracle_from_record(record)
        decoded = greedy_decode_from_z(bundle, oracle, ordered_names, z)

        state = None
        state_build_ok = False
        state_build_error = None
        measurement = None
        predicted_score = predict_score(bundle, z)

        if not decoded.final_violations:
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
            "candidate_index": candidate_index,
            "alpha": alpha,
            "coefficient_source": coefficient_source,
            "predicted_score": predicted_score,
        }
        if state_build_ok:
            sym_key = make_sym_map_key(
                {k: int(v) for k, v in decoded.sym_map.items() if isinstance(v, int)}
            )
            if sym_key in pending_measure_indices_by_key:
                pending_measure_indices_by_key[sym_key].append(len(results))
            else:
                pending_measure_indices_by_key[sym_key] = [len(results)]
                pending_measure_keys.append(sym_key)
                pending_measure_states.append(state)
                pending_measure_metas.append(measure_meta)

        results.append(
            DimInterventionRecord(
                candidate_index=candidate_index,
                alpha=alpha,
                coefficient_source=coefficient_source,
                coefficient_by_dim={int(k): float(v) for k, v in coefficient_by_dim.items()},
                predicted_score=predicted_score,
                z=z.detach().cpu().tolist(),
                params=dict(decoded.params),
                sym_map={k: int(v) for k, v in decoded.sym_map.items() if isinstance(v, int)},
                final_violations=list(decoded.final_violations),
                state_build_ok=state_build_ok,
                state_build_error=state_build_error,
                measurement=measurement,
            )
        )

    print(
        "[important-dim] unique_sym_map="
        f"{len({frozenset(rec.sym_map.items()) for rec in results})}"
    )

    _release_measurement_environment(bundle, generator=gen)

    measurement_session = _build_measurement_session(task_for_measure, measure_output_dir)
    measured_results = measure_candidates_batch(
        session=measurement_session,
        task=task_for_measure,
        states=pending_measure_states,
        metas=pending_measure_metas,
    )
    for sym_key, measurement in zip(pending_measure_keys, measured_results):
        for result_index in pending_measure_indices_by_key[sym_key]:
            results[result_index].measurement = dict(measurement)

    for grouped_record in _group_dim_records_by_sym_map(results):
        _log_grouped_intervention_result(grouped_record)

    return results, start_ctx


def save_records(records: List[DimInterventionRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def _group_dim_records_by_sym_map(records: List[DimInterventionRecord]) -> List[Dict[str, Any]]:
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


IMPORTANT_DIM_RESULT_FIELDNAMES = [
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
    "important_dim",
    "unit_coefficients",
]


def _append_important_dim_result_csv(
    *,
    csv_path: str | Path,
    record: JsonSampleRecord,
    records: List[DimInterventionRecord],
    start_ctx: StartZContext,
    best_cost: bool,
    random_z: bool,
    seed: Optional[int],
    important_dims: Sequence[int],
    use_unit_coefficients: bool,
) -> None:
    if random_z and seed is not None:
        existing_random_seeds = load_existing_random_seeds(csv_path)
        if int(seed) in existing_random_seeds:
            print(
                "[important-dim] skip csv append because random record with same seed already exists "
                f"seed={seed}"
            )
            return

    if start_ctx.encode_deterministic is True and best_cost:
        csv_path = Path(csv_path)
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_sample_id = str(row.get("sample_id") or "").strip()
                    row_deterministic = str(row.get("encode_deterministic") or "").strip().lower()
                    row_best_cost = str(row.get("best_cost") or "").strip().lower()
                    row_unit_coefficients = str(row.get("unit_coefficients") or "").strip().lower()
                    if (
                        row_sample_id == record.sample_id
                        and row_deterministic in {"1", "true", "yes", "y"}
                        and row_best_cost in {"1", "true", "yes", "y"}
                        and row_unit_coefficients
                        == ("true" if use_unit_coefficients else "")
                    ):
                        print(
                            "[important-dim] skip csv append because matching deterministic "
                            "best-cost/unit-coefficients record already exists "
                            f"sample_id={record.sample_id}"
                        )
                        return

    true_mean_cost = None if random_z else extract_true_mean_cost(record)
    important_dim_text = ",".join(str(int(dim)) for dim in important_dims)
    rows: List[Dict[str, Any]] = []
    for grouped_record in _group_dim_records_by_sym_map(records):
        record_obj = grouped_record["record"]
        measurement = record_obj.measurement or {}
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
                "important_dim": important_dim_text,
                "unit_coefficients": use_unit_coefficients,
            }
        )

    append_result_rows(csv_path, IMPORTANT_DIM_RESULT_FIELDNAMES, rows)


def _parse_important_dims(dim_sign_pairs: Optional[List[str]]) -> List[int]:
    if not dim_sign_pairs:
        return list(DEFAULT_IMPORTANT_DIMS)

    parsed: List[int] = []
    for item in dim_sign_pairs:
        text = str(item).strip()
        if ":" in text:
            dim_text, _ = text.split(":", 1)
            dim = int(dim_text)
        else:
            dim = int(text)
        if dim not in parsed:
            parsed.append(dim)
    return parsed


def run_important_dim_intervention(
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    *,
    network_info_folder: Optional[str] = None,
    device: str = "cuda",
    output: Optional[str | Path] = None,
    important_dims: Optional[Sequence[int]] = None,
    alphas: Optional[List[float]] = None,
    use_unit_coefficients: bool = False,
    random_z: bool = False,
    seed: Optional[int] = None,
    best_cost: bool = False,
    deterministic_start: bool = False,
) -> List[DimInterventionRecord]:
    record_output_path, _ = _resolve_output_layout(
        checkpoint_path=checkpoint_path,
        record_json_path=record_json_path,
        output=output,
    )
    bundle = load_bundle(
        checkpoint_path,
        network_info_folder=network_info_folder,
        device=device,
    )
    record = _select_record_from_path(record_json_path, best_cost=best_cost)

    resolved_important_dims = (
        list(DEFAULT_IMPORTANT_DIMS)
        if important_dims is None
        else [int(dim) for dim in important_dims]
    )
    records, start_ctx = build_important_dim_records(
        bundle,
        record,
        important_dims=resolved_important_dims,
        alphas=list(DEFAULT_ALPHAS if alphas is None else alphas),
        use_unit_coefficients=bool(use_unit_coefficients),
        random_z=random_z,
        seed=seed,
        output=output,
        deterministic_start=deterministic_start,
    )
    if record_output_path is not None:
        save_records(records, record_output_path)
    csv_output_path = resolve_results_csv_path(
        __file__,
        "by_important_dims",
        checkpoint_path,
        bundle.checkpoint_payload,
    )
    _append_important_dim_result_csv(
        csv_path=csv_output_path,
        record=record,
        records=records,
        start_ctx=start_ctx,
        best_cost=best_cost,
        random_z=random_z,
        seed=seed,
        important_dims=resolved_important_dims,
        use_unit_coefficients=use_unit_coefficients,
    )
    return records


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, type=str)
    p.add_argument("--record-json", required=True, type=str)
    p.add_argument("--network-info-folder", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--random",
        action="store_true",
        help="Use a standard Gaussian sample as the starting latent z instead of encoding --record-json",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when --random is enabled",
    )
    p.add_argument(
        "--best-cost",
        action="store_true",
        help="If --record-json contains multiple records, start from the one with the best recorded cost",
    )
    p.add_argument(
        "--dim-sign-pairs",
        type=str,
        nargs="*",
        default=None,
        help="Important dims to use together. Signs are ignored for compatibility. Examples: 56:+1 33:-1 15",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Intervention magnitudes. Default: 0.5 1.0",
    )
    p.add_argument(
        "--unit-coefficients", # 계수 1로 고정
        action="store_true",
        help="Use coefficient 1.0 for every important dim instead of fitting coefficients from train data",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic=True when encoding the starting latent z from --record-json",
    )
    return p


def main() -> List[DimInterventionRecord]:
    args = _build_argparser().parse_args()
    important_dims = _parse_important_dims(args.dim_sign_pairs)
    return run_important_dim_intervention(
        checkpoint_path=args.checkpoint,
        record_json_path=args.record_json,
        network_info_folder=args.network_info_folder,
        device=args.device,
        output=args.output,
        important_dims=important_dims,
        alphas=args.alphas,
        use_unit_coefficients=bool(args.unit_coefficients),
        random_z=bool(args.random),
        seed=args.seed,
        best_cost=bool(args.best_cost),
        deterministic_start=bool(args.deterministic),
    )


if __name__ == "__main__":
    main()
