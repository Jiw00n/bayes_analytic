from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from pathlib import Path
import re
from typing import Dict, List

if __package__ in (None, ""):
    import sys

    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))

    from latent_model_budget.config import build_config
    from latent_model_budget.train import train_main
else:
    from .latent_model_budget.config import build_config
    from .latent_model_budget.train import train_main


VARIANT_ENV = {
    "current": {
        "CGB_USE_DEFAULT_ENABLED_CONSTRAINTS": "0",
        "CGB_USE_RECORD_DOMAIN_PRECOMPUTE": "1",
        "CGB_USE_ORACLE_DOMAIN_SWEEP": "1",
    },
    "default_constraints": {
        "CGB_USE_DEFAULT_ENABLED_CONSTRAINTS": "1",
        "CGB_USE_RECORD_DOMAIN_PRECOMPUTE": "1",
        "CGB_USE_ORACLE_DOMAIN_SWEEP": "1",
    },
    "old_style_dataset": {
        "CGB_USE_DEFAULT_ENABLED_CONSTRAINTS": "0",
        "CGB_USE_RECORD_DOMAIN_PRECOMPUTE": "0",
        "CGB_USE_ORACLE_DOMAIN_SWEEP": "0",
    },
}


def _bool_arg(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {text}")


def _resolve_task_json(task_index: int) -> str:
    from glob import glob

    matches = []
    for path in sorted(glob("/root/work/tvm-ansor/gallery/constrained_gen/data/measured_*/*.json")):
        if f"/{int(task_index)}_" in path:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"no measured JSON found for task_index={task_index}")
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple measured JSON files matched task_index={task_index}: {matches}"
        )
    return matches[0]


def _set_env(mapping: Dict[str, str]) -> Dict[str, str | None]:
    previous: Dict[str, str | None] = {}
    for key, value in mapping.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env(previous: Dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _build_run_name(cfg) -> str:
    name = (
        f"lr{cfg.train.learning_rate}"
        f"_nce{cfg.train.lambda_nce}"
        f"_tau{cfg.train.tau_nce}"
        f"_kl{cfg.train.beta_end}"
        f"_warm{cfg.train.beta_warmup_epochs}"
    )
    if bool(cfg.train.order_nce):
        name += "_order"
    if bool(getattr(cfg.train, "nce_mu", False)):
        name += "_nce_mu"
    if bool(cfg.model.adaln):
        name += "_adaln"
    return name


def _build_run_name_from_values(
    *,
    learning_rate: float,
    lambda_nce: float,
    tau_nce: float,
    beta_end: float,
    beta_warmup_epochs: int,
    order_nce: bool,
    nce_mu: bool,
    adaln: bool,
) -> str:
    name = (
        f"lr{learning_rate}"
        f"_nce{lambda_nce}"
        f"_tau{tau_nce}"
        f"_kl{beta_end}"
        f"_warm{beta_warmup_epochs}"
    )
    if bool(order_nce):
        name += "_order"
    if bool(nce_mu):
        name += "_nce_mu"
    if bool(adaln):
        name += "_adaln"
    return name


_RUN_NAME_RE = re.compile(
    r"^lr(?P<learning_rate>[^_]+)"
    r"_nce(?P<lambda_nce>[^_]+)"
    r"_tau(?P<tau_nce>[^_]+)"
    r"_kl(?P<beta_end>[^_]+)"
    r"_warm(?P<beta_warmup_epochs>\d+)"
    r"(?P<suffix>.*)$"
)


def _parse_checkpoint_name(checkpoint_name: str) -> Dict[str, object]:
    stem = checkpoint_name[:-3] if checkpoint_name.endswith(".pt") else checkpoint_name
    match = _RUN_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"could not parse checkpoint name: {checkpoint_name}")
    suffix = match.group("suffix")
    return {
        "experiment_name": stem,
        "learning_rate": float(match.group("learning_rate")),
        "lambda_nce": float(match.group("lambda_nce")),
        "tau_nce": float(match.group("tau_nce")),
        "beta_end": float(match.group("beta_end")),
        "beta_warmup_epochs": int(match.group("beta_warmup_epochs")),
        "order_nce": "_order" in suffix,
        "nce_mu": "_nce_mu" in suffix,
        "adaln": "_adaln" in suffix,
    }


def _load_experiments_from_summary(summary_csv: str | Path) -> List[Dict[str, object]]:
    seen = set()
    experiments: List[Dict[str, object]] = []
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            checkpoint_name = str(row["checkpoint_name"]).strip()
            if not checkpoint_name or checkpoint_name in seen:
                continue
            seen.add(checkpoint_name)
            experiments.append(_parse_checkpoint_name(checkpoint_name))
    if not experiments:
        raise ValueError(f"no experiments found in summary CSV: {summary_csv}")
    return experiments


def _apply_args_to_config(cfg, args, *, json_path: str, checkpoint_dir: str) -> None:
    cfg.data.json_paths = [json_path]
    cfg.data.network_info_folder = str(args.network_info_folder)
    cfg.data.budget = bool(args.budget)

    cfg.model.adaln = bool(args.adaln)

    cfg.train.batch_size = int(args.batch_size)
    cfg.train.num_epochs = int(args.num_epochs)
    cfg.train.learning_rate = float(args.learning_rate)
    cfg.train.beta_end = float(args.beta_end)
    cfg.train.beta_warmup_epochs = int(args.beta_warmup_epochs)
    cfg.train.lambda_nce = float(args.lambda_nce)
    cfg.train.tau_nce = float(args.tau_nce)
    cfg.train.order_nce = bool(args.order_nce)
    cfg.train.nce_mu = bool(args.nce_mu)
    cfg.train.device = str(args.device)
    cfg.train.checkpoint_dir = checkpoint_dir
    cfg.train.early_stop_patience = int(args.early_stop_patience)
    cfg.train.early_stop_min_delta = float(args.early_stop_min_delta)
    cfg.train.best_metric_name = str(args.best_metric_name)
    cfg.train.best_metric_mode = str(args.best_metric_mode)
    cfg.train.precompute_candidate_masks = bool(args.precompute_candidate_masks)
    cfg.train.evaluate_train_teacher_forcing_each_epoch = bool(args.evaluate_train_tf)
    cfg.train.evaluate_cost_metrics_each_epoch = bool(args.evaluate_cost_metrics)
    cfg.train.evaluate_final_checkpoint_metrics = bool(args.evaluate_final_checkpoint_metrics)
    cfg.train.print_reconstruction_after_train = bool(args.print_reconstruction_after_train)
    cfg.train.evaluate_autoregressive_each_epoch = int(args.evaluate_autoregressive_each_epoch)

    if args.wandb_project.lower() == "none":
        cfg.wandb.project = None
    else:
        cfg.wandb.project = str(args.wandb_project)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["current", "old_style_dataset", "default_constraints"],
        choices=sorted(VARIANT_ENV.keys()),
    )
    parser.add_argument("--network-info-folder", type=str, default="/root/work/tvm-ansor/gallery/dataset/network_info_all")
    parser.add_argument("--output-root", type=str, default="/root/work/tvm-ansor/gallery/constrained_gen_budget/results/training_ablations")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-nce", type=float, default=None)
    parser.add_argument("--tau-nce", type=float, default=None)
    parser.add_argument("--beta-end", type=float, default=None)
    parser.add_argument("--beta-warmup-epochs", type=int, default=None)
    parser.add_argument("--adaln", type=_bool_arg, default=None)
    parser.add_argument("--order-nce", type=_bool_arg, default=None)
    parser.add_argument("--nce-mu", type=_bool_arg, default=None)
    parser.add_argument("--budget", type=_bool_arg, default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--best-metric-name", type=str, default="val_full_sequence_exact_match")
    parser.add_argument("--best-metric-mode", type=str, default="max")
    parser.add_argument("--precompute-candidate-masks", type=_bool_arg, default=True)
    parser.add_argument("--evaluate-train-tf", type=_bool_arg, default=True)
    parser.add_argument("--evaluate-cost-metrics", type=_bool_arg, default=True)
    parser.add_argument("--evaluate-final-checkpoint-metrics", type=_bool_arg, default=True)
    parser.add_argument("--print-reconstruction-after-train", type=_bool_arg, default=True)
    parser.add_argument("--evaluate-autoregressive-each-epoch", type=int, default=10)
    parser.add_argument("--wandb-project", type=str, default="none")
    return parser.parse_args()


def _resolve_experiments(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.summary_csv:
        return _load_experiments_from_summary(args.summary_csv)
    if args.checkpoint_name:
        return [_parse_checkpoint_name(args.checkpoint_name)]

    required = {
        "learning_rate": args.learning_rate,
        "lambda_nce": args.lambda_nce,
        "tau_nce": args.tau_nce,
        "beta_end": args.beta_end,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "adaln": args.adaln,
        "order_nce": args.order_nce,
        "nce_mu": args.nce_mu,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise SystemExit(
            "missing required arguments for single-experiment mode: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    return [
        {
            "experiment_name": None,
            "learning_rate": float(args.learning_rate),
            "lambda_nce": float(args.lambda_nce),
            "tau_nce": float(args.tau_nce),
            "beta_end": float(args.beta_end),
            "beta_warmup_epochs": int(args.beta_warmup_epochs),
            "adaln": bool(args.adaln),
            "order_nce": bool(args.order_nce),
            "nce_mu": bool(args.nce_mu),
        }
    ]


def main() -> None:
    args = parse_args()
    json_path = _resolve_task_json(args.task_index)
    experiments = _resolve_experiments(args)

    root_dir = Path(args.output_root) / str(args.task_index)
    root_dir.mkdir(parents=True, exist_ok=True)
    root_metadata = {
        "task_index": int(args.task_index),
        "json_path": json_path,
        "variants": list(args.variants),
        "summary_csv": args.summary_csv,
        "checkpoint_name": args.checkpoint_name,
        "num_experiments": len(experiments),
    }
    (root_dir / "ablation_batch_metadata.json").write_text(
        json.dumps(root_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    batch_results: List[Dict[str, object]] = []
    for experiment in experiments:
        base_cfg = build_config()
        temp_args = copy.deepcopy(args)
        temp_args.learning_rate = float(experiment["learning_rate"])
        temp_args.lambda_nce = float(experiment["lambda_nce"])
        temp_args.tau_nce = float(experiment["tau_nce"])
        temp_args.beta_end = float(experiment["beta_end"])
        temp_args.beta_warmup_epochs = int(experiment["beta_warmup_epochs"])
        temp_args.adaln = bool(experiment["adaln"])
        temp_args.order_nce = bool(experiment["order_nce"])
        temp_args.nce_mu = bool(experiment["nce_mu"])

        experiment_name = str(
            experiment["experiment_name"]
            or _build_run_name_from_values(
                learning_rate=float(temp_args.learning_rate),
                lambda_nce=float(temp_args.lambda_nce),
                tau_nce=float(temp_args.tau_nce),
                beta_end=float(temp_args.beta_end),
                beta_warmup_epochs=int(temp_args.beta_warmup_epochs),
                order_nce=bool(temp_args.order_nce),
                nce_mu=bool(temp_args.nce_mu),
                adaln=bool(temp_args.adaln),
            )
        )
        output_root = root_dir / experiment_name
        output_root.mkdir(parents=True, exist_ok=True)

        metadata = {
            "task_index": int(args.task_index),
            "json_path": json_path,
            "variants": list(args.variants),
            "experiment_name": experiment_name,
            "base_hparams": {
                "learning_rate": float(temp_args.learning_rate),
                "lambda_nce": float(temp_args.lambda_nce),
                "tau_nce": float(temp_args.tau_nce),
                "beta_end": float(temp_args.beta_end),
                "beta_warmup_epochs": int(temp_args.beta_warmup_epochs),
                "adaln": bool(temp_args.adaln),
                "order_nce": bool(temp_args.order_nce),
                "nce_mu": bool(temp_args.nce_mu),
                "budget": bool(temp_args.budget),
            },
        }
        (output_root / "ablation_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        results: List[Dict[str, object]] = []
        for variant in args.variants:
            variant_dir = output_root / variant
            checkpoint_dir = variant_dir / "checkpoints"
            variant_dir.mkdir(parents=True, exist_ok=True)

            cfg = copy.deepcopy(base_cfg)
            _apply_args_to_config(cfg, temp_args, json_path=json_path, checkpoint_dir=str(checkpoint_dir))
            run_name = _build_run_name(cfg)

            env_mapping = dict(VARIANT_ENV[variant])
            previous_env = _set_env(env_mapping)
            try:
                print("================================")
                print(f"[ablation] experiment={experiment_name}")
                print(f"[ablation] variant={variant}")
                print(f"[ablation] json_path={json_path}")
                print(f"[ablation] checkpoint_dir={checkpoint_dir}")
                print(f"[ablation] runtime_flags={env_mapping}")
                print(f"[ablation] run_name={run_name}")
                final_metrics = train_main(cfg)
                result = {
                    "experiment_name": experiment_name,
                    "variant": variant,
                    "checkpoint_dir": str(checkpoint_dir),
                    "runtime_flags": env_mapping,
                    "final_metrics": final_metrics,
                }
            except Exception as err:  # pylint: disable=broad-except
                result = {
                    "experiment_name": experiment_name,
                    "variant": variant,
                    "checkpoint_dir": str(checkpoint_dir),
                    "runtime_flags": env_mapping,
                    "error": f"{type(err).__name__}: {err}",
                }
            finally:
                _restore_env(previous_env)

            (variant_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            results.append(result)

        (output_root / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        batch_results.extend(results)

    (root_dir / "batch_results.json").write_text(
        json.dumps(batch_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (root_dir / "batch_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_name",
                "variant",
                "status",
                "checkpoint_dir",
                "best_metric_name",
                "best_metric_value",
                "error",
                "final_metrics_json",
            ],
        )
        writer.writeheader()
        for item in batch_results:
            final_metrics = item.get("final_metrics")
            best_metric_name = ""
            best_metric_value = ""
            if isinstance(final_metrics, dict):
                best_metric_name = str(final_metrics.get("best_metric_name", ""))
                best_metric_value = str(final_metrics.get("best_metric_value", ""))
            writer.writerow(
                {
                    "experiment_name": item.get("experiment_name", ""),
                    "variant": item.get("variant", ""),
                    "status": "error" if item.get("error") else "ok",
                    "checkpoint_dir": item.get("checkpoint_dir", ""),
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "error": item.get("error", ""),
                    "final_metrics_json": json.dumps(final_metrics, ensure_ascii=False, sort_keys=True)
                    if final_metrics is not None
                    else "",
                }
            )


if __name__ == "__main__":
    main()
