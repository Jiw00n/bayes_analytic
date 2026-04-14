from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

if __package__ in (None, ""):
    from modules.task_paths import clean_name
else:
    from .modules.task_paths import clean_name


def resolve_checkpoint_name(
    checkpoint_path: str | Path,
    checkpoint_payload: Dict[str, Any],
) -> str:
    checkpoint_stem = Path(checkpoint_path).stem
    if checkpoint_stem == "last":
        timestamp = checkpoint_payload.get("timestamp")
        if timestamp:
            return clean_name(str(timestamp))
    return clean_name(checkpoint_stem)


def resolve_results_csv_path(
    script_path: str | Path,
    subdir: str,
    checkpoint_path: str | Path,
    checkpoint_payload: Dict[str, Any],
) -> Path:
    root = Path(script_path).resolve().parent / "results" / subdir
    checkpoint_name = resolve_checkpoint_name(checkpoint_path, checkpoint_payload)
    return root / f"{checkpoint_name}.csv"


def serialize_csv_value(value: Any) -> str:
    if value is None or value is False:
        return ""
    if value is True:
        return "true"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    return str(value)


def extract_true_mean_cost(record: Any) -> Optional[float]:
    payload = getattr(record, "raw", None)
    if not isinstance(payload, dict):
        return None

    raw_result = payload.get("r")
    if isinstance(raw_result, list) and len(raw_result) >= 2:
        costs = raw_result[0]
        error_no = raw_result[1]
        try:
            if int(error_no) != 0:
                return None
        except (TypeError, ValueError):
            return None
        if isinstance(costs, list) and costs:
            try:
                float_costs = [float(x) for x in costs]
            except (TypeError, ValueError):
                return None
            if float_costs:
                return -math.log(float(sum(float_costs) / len(float_costs)))

    for key in ("cost", "cost_target"):
        value = payload.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value

    return None


def load_existing_deterministic_sample_ids(csv_path: str | Path) -> Set[str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()

    sample_ids: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            deterministic = str(row.get("encode_deterministic") or "").strip().lower()
            if sample_id and deterministic in {"1", "true", "yes", "y"}:
                sample_ids.add(sample_id)
    return sample_ids


def load_existing_random_seeds(csv_path: str | Path) -> Set[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()

    seeds: Set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_flag = str(row.get("random") or "").strip().lower()
            seed_text = str(row.get("seed") or "").strip()
            if random_flag not in {"1", "true", "yes", "y"}:
                continue
            if not seed_text:
                continue
            try:
                seeds.add(int(seed_text))
            except (TypeError, ValueError):
                continue
    return seeds


def append_result_rows(
    csv_path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> None:
    csv_path = Path(csv_path)
    rows = list(rows)
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: serialize_csv_value(row.get(name)) for name in fieldnames})


def make_sym_map_key(sym_map: Dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(name), int(value)) for name, value in sym_map.items()))
