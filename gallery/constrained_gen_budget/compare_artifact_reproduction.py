from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_checkpoint_artifacts(checkpoint_path: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    tokenizer_state = payload.get("tokenizer_state") or payload.get("tokenizer") or {}
    cfg = payload.get("config", {})
    return {
        "checkpoint_path": str(checkpoint_path),
        "config": cfg,
        "tokenizer": {
            "id_to_token": list(tokenizer_state.get("id_to_token", [])),
            "id_to_var": list(tokenizer_state.get("id_to_var", [])),
        },
    }


def _build_fresh_artifacts(
    *,
    code_root: Path,
    json_path: Path,
    network_info_folder: Path,
    budget: bool,
    precompute_candidate_masks: bool,
) -> Dict[str, Any]:
    inline = r"""
import json
from latent_model_budget.config import build_config
from latent_model_budget.train import build_everything

cfg = build_config()
cfg.data.json_paths = [JSON_PATH]
cfg.data.network_info_folder = NETWORK_INFO_FOLDER
cfg.data.budget = BUDGET
cfg.train.precompute_candidate_masks = PRECOMPUTE

registry, bundle, tokenizer, model = build_everything(cfg)

result = {
    "train_count": len(bundle.train_dataset.samples),
    "val_count": len(bundle.val_dataset.samples),
    "test_count": len(bundle.test_dataset.samples),
    "first_train_sample_id": None if not bundle.train_records else bundle.train_records[0].sample_id,
    "first_val_sample_id": None if not bundle.val_records else bundle.val_records[0].sample_id,
    "tokenizer": {
        "id_to_token": list(tokenizer.id_to_token),
        "id_to_var": list(tokenizer.id_to_var),
    },
}
print(json.dumps(result, ensure_ascii=False))
"""
    inline = (
        inline.replace("JSON_PATH", repr(str(json_path)))
        .replace("NETWORK_INFO_FOLDER", repr(str(network_info_folder)))
        .replace("BUDGET", "True" if budget else "False")
        .replace("PRECOMPUTE", "True" if precompute_candidate_masks else "False")
    )
    env = dict(os.environ)
    tvm_home = env.get("TVM_HOME", "/root/work/tvm-ansor")
    py_paths = [str(code_root), f"{tvm_home}/python"]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)
    env.setdefault("TVM_LIBRARY_PATH", f"{tvm_home}/build-release")
    cmd = [sys.executable, "-c", inline]
    proc = subprocess.run(
        cmd,
        cwd=str(code_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fresh build failed for {code_root}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"fresh build produced no JSON output for {code_root}")
    return json.loads(lines[-1])


def _diff_lists(left: List[str], right: List[str]) -> Dict[str, Any]:
    return {
        "equal": left == right,
        "left_only": sorted(set(left) - set(right)),
        "right_only": sorted(set(right) - set(left)),
        "left_len": len(left),
        "right_len": len(right),
    }


def _summarize_side(
    *,
    label: str,
    code_root: Path,
    checkpoint_path: Path,
    json_path: Path,
    network_info_folder: Path,
    budget: bool,
    precompute_candidate_masks: bool,
) -> Dict[str, Any]:
    checkpoint = _load_checkpoint_artifacts(checkpoint_path)
    fresh = _build_fresh_artifacts(
        code_root=code_root,
        json_path=json_path,
        network_info_folder=network_info_folder,
        budget=budget,
        precompute_candidate_masks=precompute_candidate_masks,
    )
    ck_tokens = checkpoint["tokenizer"]["id_to_token"]
    ck_vars = checkpoint["tokenizer"]["id_to_var"]
    fresh_tokens = fresh["tokenizer"]["id_to_token"]
    fresh_vars = fresh["tokenizer"]["id_to_var"]
    return {
        "label": label,
        "code_root": str(code_root),
        "checkpoint_path": str(checkpoint_path),
        "json_path": str(json_path),
        "budget": bool(budget),
        "precompute_candidate_masks": bool(precompute_candidate_masks),
        "checkpoint": {
            "num_tokens": len(ck_tokens),
            "num_vars": len(ck_vars),
            "tokens": ck_tokens,
            "vars": ck_vars,
        },
        "fresh": {
            "num_tokens": len(fresh_tokens),
            "num_vars": len(fresh_vars),
            "tokens": fresh_tokens,
            "vars": fresh_vars,
            "train_count": fresh["train_count"],
            "val_count": fresh["val_count"],
            "test_count": fresh["test_count"],
            "first_train_sample_id": fresh["first_train_sample_id"],
            "first_val_sample_id": fresh["first_val_sample_id"],
        },
        "reproduction": {
            "tokens": _diff_lists(ck_tokens, fresh_tokens),
            "vars": _diff_lists(ck_vars, fresh_vars),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left-code-root",
        type=str,
        default="/root/work/tvm-ansor/gallery/constrained_gen_budget_old",
    )
    parser.add_argument(
        "--left-checkpoint",
        type=str,
        default="/root/work/tvm-ansor/gallery/constrained_gen_budget/old/checkpoints/grid_search/lr0.0005_nce0.1_tau0.2_kl0.002_warm20.pt",
    )
    parser.add_argument(
        "--right-code-root",
        type=str,
        default="/root/work/tvm-ansor/gallery/constrained_gen_budget",
    )
    parser.add_argument(
        "--right-checkpoint",
        type=str,
        default="/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints_all/1490/grid_search/lr0.0005_nce0.1_tau0.2_kl0.002_warm20.pt",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json",
    )
    parser.add_argument(
        "--network-info-folder",
        type=str,
        default="/root/work/tvm-ansor/gallery/dataset/network_info_all",
    )
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--precompute-candidate-masks", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    network_info_folder = Path(args.network_info_folder)

    left = _summarize_side(
        label="left",
        code_root=Path(args.left_code_root),
        checkpoint_path=Path(args.left_checkpoint),
        json_path=json_path,
        network_info_folder=network_info_folder,
        budget=bool(args.budget),
        precompute_candidate_masks=bool(args.precompute_candidate_masks),
    )
    right = _summarize_side(
        label="right",
        code_root=Path(args.right_code_root),
        checkpoint_path=Path(args.right_checkpoint),
        json_path=json_path,
        network_info_folder=network_info_folder,
        budget=bool(args.budget),
        precompute_candidate_masks=bool(args.precompute_candidate_masks),
    )

    report = {
        "note": (
            "Compares historical checkpoint tokenizer artifacts against fresh bundle artifacts "
            "built under the specified code roots."
        ),
        "left": left,
        "right": right,
        "cross_checkpoint": {
            "tokens": _diff_lists(left["checkpoint"]["tokens"], right["checkpoint"]["tokens"]),
            "vars": _diff_lists(left["checkpoint"]["vars"], right["checkpoint"]["vars"]),
        },
        "cross_fresh": {
            "tokens": _diff_lists(left["fresh"]["tokens"], right["fresh"]["tokens"]),
            "vars": _diff_lists(left["fresh"]["vars"], right["fresh"]["vars"]),
        },
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[saved] {out}")
    print(text)


if __name__ == "__main__":
    main()
