from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from tvm import auto_scheduler

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))
    from latent_model_projector.adapter import GeneratorRegistry, JsonSampleRecord, LegalPrefixOracle, load_json_sample
    from latent_model_projector.model import LatentParamVAE
    from latent_model_projector.tokenizer import ParamTokenizer
    from modules.task_paths import clean_name, get_measure_record_filename
else:
    from .latent_model_projector.adapter import GeneratorRegistry, JsonSampleRecord, LegalPrefixOracle, load_json_sample
    from .latent_model_projector.model import LatentParamVAE
    from .latent_model_projector.tokenizer import ParamTokenizer
    from .modules.task_paths import clean_name, get_measure_record_filename


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


def _build_measurement_session(task: Any, measure_output_dir: Optional[str] = None) -> MeasurementSession:
    output_path = str(get_measure_record_filename(task, task.target, output_dir=measure_output_dir))
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
) -> Dict[str, Any]:
    """단일 MeasureResult를 요약 딕셔너리로 변환한다."""
    error_no = int(result.error_no)
    costs = [float(x) for x in result.costs]
    mean_cost = float(sum(costs) / len(costs)) if costs else None
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
        measure_candidate(result=result, meta=meta, session=session)
        for result, meta in zip(results, metas)
    ]


def log_candidate_result(result: Dict[str, Any]) -> None:
    """사람이 읽기 쉬운 한 줄 진행 로그를 출력한다."""
    step_index = result["step_index"]
    alpha = result["alpha"]
    predicted_score = result["predicted_score"]

    if result["final_violations"]:
        print(
            f"[latent-walk] step={step_index} alpha={alpha:.4f} "
            f"pred={predicted_score:.6f} status=violated "
            f"violations={len(result['final_violations'])}"
        )
        return

    if not result["state_build_ok"]:
        print(
            f"[latent-walk] step={step_index} alpha={alpha:.4f} "
            f"pred={predicted_score:.6f} status=state_build_failed "
            f"error={result['state_build_error']}"
        )
        return

    measurement = result.get("measurement") or {}
    if not measurement.get("ok"):
        print(
            f"[latent-walk] step={step_index} alpha={alpha:.4f} "
            f"pred={predicted_score:.6f} status=measure_failed "
            f"error={measurement.get('error')}"
        )
        return

    if not measurement.get("usable_measurement"):
        print(
            f"[latent-walk] step={step_index} alpha={alpha:.4f} "
            f"pred={predicted_score:.6f} status=measure_error "
            f"error_no={measurement.get('error_no')} "
            f"error_msg={measurement.get('error_msg')}"
        )
        return

    mean_cost = measurement.get("mean_cost")
    mean_cost_text = "n/a" if mean_cost is None else f"{mean_cost:.9f}"
    print(
        f"[latent-walk] step={step_index} alpha={alpha:.4f} "
        f"pred={predicted_score:.6f} status=ok "
        f"mean_cost={mean_cost_text} "
        f"log={measurement.get('measure_record_path')}"
    )


def _default_walk_output_path(
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    output_dir: str | Path,
) -> Path:
    checkpoint_stem = clean_name(Path(checkpoint_path).stem)
    record_stem = clean_name(Path(record_json_path).stem)
    return Path(output_dir) / "walk_records" / f"{checkpoint_stem}__{record_stem}.jsonl"


def _resolve_output_layout(
    *,
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    output: Optional[str | Path],
) -> tuple[Optional[Path], Optional[str]]:
    if output is None:
        return None, None

    output_root = Path(output)
    walk_output_path = _default_walk_output_path(checkpoint_path, record_json_path, output_root)
    measure_output_dir = str(output_root / "measure_records")
    return walk_output_path, measure_output_dir


def _resolve_measure_output_dir(output: Optional[str | Path]) -> Optional[str]:
    if output is None:
        return None
    return str(Path(output) / "measure_records")


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


@dataclass
class DecodeRecord:
    params: Dict[str, int]
    sym_map: Dict[str, int]
    final_violations: List[str]



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
    if latent_cost_ridge is not None and "weight" in latent_cost_ridge:
        cost_weight = latent_cost_ridge["weight"].detach().to(
            dtype=torch.float32,
            device=torch_device,
        )
        cost_bias = float(latent_cost_ridge.get("bias", 0.0))
        cost_source = "latent_cost_ridge"
    elif use_latent_gradient:
        cost_source = "cost_head_gradient"

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
def encode_record_to_z(
    bundle: LoadedBundle,
    record: JsonSampleRecord,
) -> tuple[torch.Tensor, Any, List[str], List[int]]:
    gen = bundle.registry.get_generator_from_record(record)
    ordered_names = list(gen.get_full_var_order_entries()["param_order"])
    ordered_values = [int(record.params[name]) for name in ordered_names]

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
    _, _, z, _ = bundle.model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=False)
    return z[0].detach().clone(), gen, ordered_names, ordered_values



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

    for step_idx, var_name in enumerate(ordered_names):
        del step_idx
        candidate_values = list(oracle.candidate_values(var_name))
        if not candidate_values:
            raise RuntimeError(f"No legal candidates returned for {var_name}")

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



def predict_score(bundle: LoadedBundle, z: torch.Tensor) -> float:
    if bundle.cost_source == "cost_head_gradient":
        z = z.to(device=bundle.device, dtype=torch.float32).view(1, -1)
        return float(bundle.model.cost_head(z).squeeze(-1).item())
    if bundle.cost_weight is None:
        raise RuntimeError("No stored cost vector in checkpoint. Re-run with --latent-gradient.")
    z = z.to(device=bundle.device, dtype=torch.float32)
    return float((z @ bundle.cost_weight + bundle.cost_bias).item())


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
    output: Optional[str | Path] = None,
) -> List[WalkRecord]:
    z0, gen, ordered_names, _ = encode_record_to_z(bundle, record)
    measure_output_dir = _resolve_measure_output_dir(output)
    measurement_session = _build_measurement_session(gen._task, measure_output_dir)
    walk_direction = compute_walk_direction(bundle, z0)
    shifted_zs = make_shifted_zs(
        z0,
        walk_direction,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
    )

    results: List[WalkRecord] = []
    pending_measure_indices: List[int] = []
    pending_measure_states: List[Any] = []
    pending_measure_metas: List[Dict[str, Any]] = []
    for step_index, alpha, z in shifted_zs:
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
            "step_index": step_index,
            "alpha": alpha,
            "predicted_score": predicted_score,
        }
        if state_build_ok:
            pending_measure_indices.append(len(results))
            pending_measure_states.append(state)
            pending_measure_metas.append(measure_meta)

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
        )
        results.append(walk_record)

    print(f"Unique sym_map entries across walk: {len({frozenset(rec.sym_map.items()) for rec in results})}")

    measured_results = measure_candidates_batch(
        session=measurement_session,
        task=gen._task,
        states=pending_measure_states,
        metas=pending_measure_metas,
    )
    for result_index, measurement in zip(pending_measure_indices, measured_results):
        results[result_index].measurement = measurement

    for walk_record in results:
        log_candidate_result(asdict(walk_record))

    return results



def save_walk_records(records: List[WalkRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")



def run_latent_walk(
    checkpoint_path: str | Path,
    record_json_path: str | Path,
    *,
    network_info_folder: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 8,
    step_size: float = 0.25,
    normalize_direction: bool = True,
    output: Optional[str | Path] = None,
    latent_gradient: bool = False,
) -> List[WalkRecord]:
    walk_output_path, _ = _resolve_output_layout(
        checkpoint_path=checkpoint_path,
        record_json_path=record_json_path,
        output=output,
    )
    bundle = load_bundle(
        checkpoint_path,
        network_info_folder=network_info_folder,
        device=device,
        use_latent_gradient=latent_gradient,
    )
    if bundle.cost_source == "missing_cost_vector":
        print("[latent-walk] checkpoint에 저장된 cost vector가 없습니다.")
        return []
    record = load_json_sample(record_json_path)
    records = build_walk_records(
        bundle,
        record,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
        output=output,
    )
    if walk_output_path is not None:
        save_walk_records(records, walk_output_path)
    return records



def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--record-json", required=True, type=str)
    p.add_argument("--network-info-folder", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--no-normalize-direction", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--latent-gradient", action="store_true")
    return p



def main() -> List[WalkRecord]:
    args = _build_argparser().parse_args()
    return run_latent_walk(
        checkpoint_path=args.checkpoint,
        record_json_path=args.record_json,
        network_info_folder=args.network_info_folder,
        device=args.device,
        num_steps=args.num_steps,
        step_size=args.step_size,
        normalize_direction=not args.no_normalize_direction,
        output=args.output,
        latent_gradient=args.latent_gradient,
    )


if __name__ == "__main__":
    main()
