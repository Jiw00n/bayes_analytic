#!/bin/bash

set -euo pipefail

source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH="$TVM_HOME/python:/root/work/tvm-ansor/gallery/constrained_gen_budget"
export TVM_LIBRARY_PATH="$TVM_HOME/build-release"

ROOT_DIR="/root/work/tvm-ansor/gallery/constrained_gen_budget"
COMPARE_SCRIPT="$ROOT_DIR/compare_latent_walk_checkpoints.py"

OLD_DIR_DEFAULT="$ROOT_DIR/old/checkpoints/grid_search"
NEW_DIR_DEFAULT="$ROOT_DIR/checkpoints_all/1490/grid_search"
OUTPUT_ROOT_DEFAULT="$ROOT_DIR/results/checkpoint_compare_1490"

OLD_DIR="${OLD_DIR:-$OLD_DIR_DEFAULT}"
NEW_DIR="${NEW_DIR:-$NEW_DIR_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
DEVICE="${DEVICE:-cuda}"
NUM_STEPS="${NUM_STEPS:-30}"
STEP_SIZE="${STEP_SIZE:-0.25}"
GOLD_TRACE_STEPS="${GOLD_TRACE_STEPS:-20}"
BEST_COST="${BEST_COST:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"
NORMALIZE_DIRECTION="${NORMALIZE_DIRECTION:-1}"
RECORD_JSON="${RECORD_JSON:-}"
NETWORK_INFO_FOLDER="${NETWORK_INFO_FOLDER:-}"
LIMIT_COMMON="${LIMIT_COMMON:-0}"
EXCLUDE_NAMES="${EXCLUDE_NAMES:-best.pt,last.pt}"
CGB_USE_DEFAULT_ENABLED_CONSTRAINTS="${CGB_USE_DEFAULT_ENABLED_CONSTRAINTS:-0}"

mkdir -p "$OUTPUT_ROOT"

COMMON_LIST="$OUTPUT_ROOT/common_checkpoints.txt"
PAIR_JSON="$OUTPUT_ROOT/pairs.json"
OLD_ONLY_LIST="$OUTPUT_ROOT/old_only_checkpoints.txt"
NEW_ONLY_LIST="$OUTPUT_ROOT/new_only_checkpoints.txt"
SKIPPED_MISMATCH_LIST="$OUTPUT_ROOT/skipped_task_mismatch.txt"
REPORT_DIR="$OUTPUT_ROOT/reports"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

python - <<'PY' "$OLD_DIR" "$NEW_DIR" "$COMMON_LIST" "$PAIR_JSON" "$OLD_ONLY_LIST" "$NEW_ONLY_LIST" "$SKIPPED_MISMATCH_LIST" "$LIMIT_COMMON" "$EXCLUDE_NAMES"
from pathlib import Path
import json
import sys
import torch

old_dir = Path(sys.argv[1])
new_dir = Path(sys.argv[2])
common_path = Path(sys.argv[3])
pair_json_path = Path(sys.argv[4])
old_only_path = Path(sys.argv[5])
new_only_path = Path(sys.argv[6])
skipped_mismatch_path = Path(sys.argv[7])
limit_common = int(sys.argv[8])
exclude_names = {name.strip() for name in sys.argv[9].split(",") if name.strip()}


def checkpoint_task_signature(path: Path):
    payload = torch.load(path, map_location="cpu")
    cfg = payload.get("config", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    json_paths = data_cfg.get("json_paths", []) or []
    normalized_paths = [str(p) for p in json_paths if isinstance(p, str)]
    first_name = None
    task_prefix = None
    if normalized_paths:
        first_name = Path(normalized_paths[0]).name
        if "_" in first_name:
            task_prefix = first_name.split("_", 1)[0]
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    return {
        "json_paths": normalized_paths,
        "first_json_name": first_name,
        "task_prefix": task_prefix,
        "adaln": model_cfg.get("adaln"),
        "learning_rate": train_cfg.get("learning_rate"),
        "lambda_nce": train_cfg.get("lambda_nce"),
        "order_nce": train_cfg.get("order_nce"),
        "nce_mu": train_cfg.get("nce_mu"),
    }

old_names = sorted(p.name for p in old_dir.glob("*.pt"))
new_names = sorted(p.name for p in new_dir.glob("*.pt"))
old_set = set(old_names)
new_set = set(new_names)
common_names = sorted((old_set & new_set) - exclude_names)
old_only = sorted(old_set - new_set)
new_only = sorted(new_set - old_set)

pairs = []
skipped_mismatch = []
for name in common_names:
    old_path = old_dir / name
    new_path = new_dir / name
    old_sig = checkpoint_task_signature(old_path)
    new_sig = checkpoint_task_signature(new_path)
    if old_sig["task_prefix"] != new_sig["task_prefix"]:
        skipped_mismatch.append(
            {
                "checkpoint_name": name,
                "old_path": str(old_path),
                "new_path": str(new_path),
                "old_task_prefix": old_sig["task_prefix"],
                "new_task_prefix": new_sig["task_prefix"],
                "old_first_json_name": old_sig["first_json_name"],
                "new_first_json_name": new_sig["first_json_name"],
            }
        )
        continue
    pairs.append(
        {
            "checkpoint_name": name,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "task_prefix": old_sig["task_prefix"],
            "old_signature": old_sig,
            "new_signature": new_sig,
        }
    )

if limit_common > 0:
    pairs = pairs[:limit_common]
common = [item["checkpoint_name"] for item in pairs]

common_path.write_text("\n".join(common) + ("\n" if common else ""), encoding="utf-8")
old_only_path.write_text("\n".join(old_only) + ("\n" if old_only else ""), encoding="utf-8")
new_only_path.write_text("\n".join(new_only) + ("\n" if new_only else ""), encoding="utf-8")
pair_json_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
skipped_mismatch_path.write_text(
    json.dumps(skipped_mismatch, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(
    f"common={len(common)} old_only={len(old_only)} new_only={len(new_only)} "
    f"skipped_task_mismatch={len(skipped_mismatch)}"
)
PY

mapfile -t COMMON_CHECKPOINTS < "$COMMON_LIST"

if [[ "${#COMMON_CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No common checkpoint basenames found between:"
    echo "  OLD_DIR=$OLD_DIR"
    echo "  NEW_DIR=$NEW_DIR"
    exit 1
fi

echo "Comparing ${#COMMON_CHECKPOINTS[@]} common checkpoints"
echo "  OLD_DIR=$OLD_DIR"
echo "  NEW_DIR=$NEW_DIR"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  CGB_USE_DEFAULT_ENABLED_CONSTRAINTS=$CGB_USE_DEFAULT_ENABLED_CONSTRAINTS"

for checkpoint_name in "${COMMON_CHECKPOINTS[@]}"; do
    old_path="$OLD_DIR/$checkpoint_name"
    new_path="$NEW_DIR/$checkpoint_name"
    report_path="$REPORT_DIR/${checkpoint_name%.pt}.json"
    log_path="$LOG_DIR/${checkpoint_name%.pt}.log"

    echo "================================"
    echo "Comparing: $checkpoint_name"

    cmd=(
        python "$COMPARE_SCRIPT"
        --checkpoint-a "$old_path"
        --checkpoint-b "$new_path"
        --device "$DEVICE"
        --num-steps "$NUM_STEPS"
        --step-size "$STEP_SIZE"
        --gold-trace-steps "$GOLD_TRACE_STEPS"
        --output "$report_path"
    )

    if [[ "$BEST_COST" == "1" ]]; then
        cmd+=(--best-cost)
    fi
    if [[ "$DETERMINISTIC" == "1" ]]; then
        cmd+=(--deterministic)
    fi
    if [[ "$NORMALIZE_DIRECTION" != "1" ]]; then
        cmd+=(--no-normalize-direction)
    fi
    if [[ -n "$RECORD_JSON" ]]; then
        cmd+=(--record-json "$RECORD_JSON")
    fi
    if [[ -n "$NETWORK_INFO_FOLDER" ]]; then
        cmd+=(--network-info-folder "$NETWORK_INFO_FOLDER")
    fi

    if "${cmd[@]}" >"$log_path" 2>&1; then
        echo "  saved report: $report_path"
        echo "  saved log:    $log_path"
    else
        echo "Comparison failed for $checkpoint_name"
        echo "  log: $log_path"
        tail -n 120 "$log_path" || true
        exit 1
    fi
done

SUMMARY_JSON="$OUTPUT_ROOT/summary.json"
SUMMARY_CSV="$OUTPUT_ROOT/summary.csv"
METADATA_JSON="$OUTPUT_ROOT/metadata.json"

python - <<'PY' "$REPORT_DIR" "$SUMMARY_JSON" "$SUMMARY_CSV" "$METADATA_JSON" "$OLD_DIR" "$NEW_DIR" "$DEVICE" "$NUM_STEPS" "$STEP_SIZE" "$BEST_COST" "$DETERMINISTIC" "$NORMALIZE_DIRECTION" "$RECORD_JSON" "$NETWORK_INFO_FOLDER" "$CGB_USE_DEFAULT_ENABLED_CONSTRAINTS"
from pathlib import Path
import csv
import json
import sys

report_dir = Path(sys.argv[1])
summary_json_path = Path(sys.argv[2])
summary_csv_path = Path(sys.argv[3])
metadata_json_path = Path(sys.argv[4])

metadata = {
    "old_dir": sys.argv[5],
    "new_dir": sys.argv[6],
    "device": sys.argv[7],
    "num_steps": int(sys.argv[8]),
    "step_size": float(sys.argv[9]),
    "best_cost": sys.argv[10] == "1",
    "deterministic": sys.argv[11] == "1",
    "normalize_direction": sys.argv[12] == "1",
    "record_json": sys.argv[13] or None,
    "network_info_folder": sys.argv[14] or None,
    "runtime_flags": {
        "CGB_USE_DEFAULT_ENABLED_CONSTRAINTS": sys.argv[15] == "1",
    },
}

rows = []
for report_path in sorted(report_dir.glob("*.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = report["comparison"]
    left = report["left"]
    right = report["right"]
    row = {
        "checkpoint_name": left["checkpoint_name"],
        "same_record_json": bool(comparison["same_record_json"]),
        "same_sample_id": bool(comparison["same_sample_id"]),
        "order_equal": bool(comparison["order_equal"]),
        "first_order_difference_index": None,
        "first_order_difference_left": None,
        "first_order_difference_right": None,
        "old_num_tokens": int(comparison["tokenizer_num_tokens"]["left"]),
        "new_num_tokens": int(comparison["tokenizer_num_tokens"]["right"]),
        "old_num_vars": int(comparison["tokenizer_num_vars"]["left"]),
        "new_num_vars": int(comparison["tokenizer_num_vars"]["right"]),
        "old_unique_sym_maps": int(comparison["unique_sym_maps"]["left"]),
        "new_unique_sym_maps": int(comparison["unique_sym_maps"]["right"]),
        "shared_unique_sym_maps": int(comparison["unique_sym_maps"]["shared"]),
        "first_alpha_where_sym_map_differs": comparison["first_alpha_where_sym_map_differs"],
        "z0_cosine": comparison["latent_similarity"]["z0"]["cosine"],
        "direction_cosine": comparison["latent_similarity"]["direction"]["cosine"],
        "old_gold_failure_step": None,
        "old_gold_failure_var": None,
        "new_gold_failure_step": None,
        "new_gold_failure_var": None,
        "old_adaln": bool(left["config_summary"]["adaln"]),
        "new_adaln": bool(right["config_summary"]["adaln"]),
        "old_lr": left["config_summary"]["learning_rate"],
        "new_lr": right["config_summary"]["learning_rate"],
        "old_lambda_nce": left["config_summary"]["lambda_nce"],
        "new_lambda_nce": right["config_summary"]["lambda_nce"],
        "old_order_nce": bool(left["config_summary"]["order_nce"]),
        "new_order_nce": bool(right["config_summary"]["order_nce"]),
        "old_nce_mu": bool(left["config_summary"]["nce_mu"]),
        "new_nce_mu": bool(right["config_summary"]["nce_mu"]),
        "report_path": str(report_path),
    }
    first_order_difference = comparison.get("first_order_difference")
    if first_order_difference is not None:
        row["first_order_difference_index"] = first_order_difference.get("index")
        row["first_order_difference_left"] = first_order_difference.get("left")
        row["first_order_difference_right"] = first_order_difference.get("right")
    old_gold_failure = comparison["gold_path"]["left_first_failure"]
    if old_gold_failure is not None:
        row["old_gold_failure_step"] = old_gold_failure.get("step")
        row["old_gold_failure_var"] = old_gold_failure.get("var_name")
    new_gold_failure = comparison["gold_path"]["right_first_failure"]
    if new_gold_failure is not None:
        row["new_gold_failure_step"] = new_gold_failure.get("step")
        row["new_gold_failure_var"] = new_gold_failure.get("var_name")
    rows.append(row)

summary = {
    "metadata": metadata,
    "num_reports": len(rows),
    "rows": rows,
}

summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

fieldnames = [
    "checkpoint_name",
    "same_record_json",
    "same_sample_id",
    "order_equal",
    "first_order_difference_index",
    "first_order_difference_left",
    "first_order_difference_right",
    "old_num_tokens",
    "new_num_tokens",
    "old_num_vars",
    "new_num_vars",
    "old_unique_sym_maps",
    "new_unique_sym_maps",
    "shared_unique_sym_maps",
    "first_alpha_where_sym_map_differs",
    "z0_cosine",
    "direction_cosine",
    "old_gold_failure_step",
    "old_gold_failure_var",
    "new_gold_failure_step",
    "new_gold_failure_var",
    "old_adaln",
    "new_adaln",
    "old_lr",
    "new_lr",
    "old_lambda_nce",
    "new_lambda_nce",
    "old_order_nce",
    "new_order_nce",
    "old_nce_mu",
    "new_nce_mu",
    "report_path",
]
with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

metadata_json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[saved] {summary_json_path}")
print(f"[saved] {summary_csv_path}")
print(f"[saved] {metadata_json_path}")
PY

echo "================================"
echo "Finished."
echo "  common checkpoints: ${#COMMON_CHECKPOINTS[@]}"
echo "  summary json: $SUMMARY_JSON"
echo "  summary csv:  $SUMMARY_CSV"
echo "  reports dir:   $REPORT_DIR"
echo "  logs dir:      $LOG_DIR"
