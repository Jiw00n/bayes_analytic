"""Pre-seed per-task measurement-lookup files from existing measure-record
JSONs.

Adapted from ``constrained_gen_budget_v1.5_ori/build_measurement_lookup.py``.
The differences vs. the upstream variant are:

- Input is a directory of TVM ``RecordToFile`` outputs (one file per task)
  instead of a gallery scan. Each input file's filename is expected to start
  with ``{task_index}_`` so we can resolve the matching generator without an
  explicit task argument.
- Output entries include ``workload_key`` because
  ``latent_model_budget/train.py`` now keys its measurement cache on
  ``(workload_key, sym_map)`` to prevent cross-task contamination of the
  lookup. Records without a workload_key would be dropped at load time, so
  emitting it is mandatory.
- One output file per input file, named identically to the input but with a
  ``.jsonl`` extension and placed in a sibling ``lookup_sym_maps/`` directory
  (default — both paths overridable).

Usage
-----
    python build_measurement_lookup.py \\
        [--input-dir  <measure_records dir>] \\
        [--output-dir <lookup_sym_maps dir>] \\
        [--network-info-folder ...] \\
        [--include-budget]

Defaults match the layout produced by training:
``checkpoints_all/<family>/measure_records/`` →
``checkpoints_all/<family>/lookup_sym_maps/``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
GALLERY_ROOT = HERE.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget.adapter import (  # noqa: E402
    GeneratorRegistry,
    JsonSampleRecord,
    _augment_record_budget_params,
    _normalize_generator_signature,
    cost_label_to_raw,
    load_json_samples,
)
from latent_model_budget.config import DEFAULT_NETWORK_INFO_FOLDER  # noqa: E402
from result_csv_utils import make_sym_map_key  # noqa: E402


SymKey = Tuple[Tuple[str, int], ...]

# Default I/O paths line up with the on-disk layout produced by training:
#   checkpoints_all/{family}/measure_records/{task_index}_*.json
#   checkpoints_all/{family}/lookup_sym_maps/{task_index}_*.jsonl
DEFAULT_INPUT_DIR = (
    HERE
    / "checkpoints_all"
    / "nn_contrib_conv2d_winograd_without_weight_transform"
    / "measure_records"
)
DEFAULT_OUTPUT_DIR = (
    HERE
    / "checkpoints_all"
    / "nn_contrib_conv2d_winograd_without_weight_transform"
    / "lookup_sym_maps"
)


def _parse_task_index_from_filename(path: Path) -> Optional[int]:
    """Filenames are ``{task_index}_<base_key>.json`` (the layout ``train.py``
    writes via ``_resolve_walk_measure_record_path``). Returns the prefix as
    int, or ``None`` when the filename doesn't match."""
    m = re.match(r"^(\d+)_", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


class _SketchMatcher:
    """Resolves ``record.param_signature`` → cached sketch-based generator.

    Mirrors the helper from
    ``constrained_gen_budget_v1.5_ori/build_measurement_lookup.py``. Rebuilding
    the generator per record is the slow path; routing every record through
    the sketch-index cache means one generator per unique sketch signature."""

    def __init__(self, registry: GeneratorRegistry, task_index: int):
        self.registry = registry
        self.task_index = int(task_index)
        self.task = registry._resolve_task(task_index=self.task_index)
        self.workload_key, self.target_kind = registry._task_signature(self.task)
        self._sig_to_sketch_index: Dict[Tuple[str, ...], int] = {}
        self._generators: Dict[int, object] = {}
        self._sketches = registry._get_sketches_for_task(self.task)
        for sketch_index in range(len(self._sketches)):
            gen = self._get_generator(sketch_index)
            sig = tuple(gen.s.sym_map.keys())
            self._sig_to_sketch_index.setdefault(sig, sketch_index)

    def _get_generator(self, sketch_index: int):
        cached = self._generators.get(int(sketch_index))
        if cached is not None:
            return cached
        gen = self.registry.get_generator(
            workload_key=self.workload_key,
            target_kind=self.target_kind,
            sketch_index=int(sketch_index),
        )
        self._generators[int(sketch_index)] = gen
        return gen

    def resolve(self, record: JsonSampleRecord):
        if not record.param_signature:
            raise ValueError(f"record {record.sample_id} has no param_signature")
        normalized = _normalize_generator_signature(record.param_signature)
        sketch_index = self._sig_to_sketch_index.get(normalized)
        if sketch_index is None:
            for idx in range(len(self._sketches)):
                gen = self._get_generator(idx)
                if tuple(gen.s.sym_map.keys()) == normalized:
                    sketch_index = idx
                    self._sig_to_sketch_index[normalized] = idx
                    break
        if sketch_index is None:
            raise KeyError(
                f"No sketch matches record signature: {normalized!r}"
            )
        return self._get_generator(sketch_index)


def _is_budget_name(name: str) -> bool:
    return name.startswith("thread_budget") or name.startswith("vthread_budget")


def _compute_sym_key(
    record: JsonSampleRecord,
    matcher: _SketchMatcher,
    *,
    include_budget: bool,
) -> Optional[SymKey]:
    """Same shape ``build_walk_records`` produces at walk time: sp_/ur_ entries
    only (plus budget entries when ``include_budget`` is set). The walk's
    sym_key is a sorted ``(name, value)`` tuple; the on-disk lookup stores
    ``sym_map`` as a dict and ``train.py`` reconstructs the tuple via
    ``make_task_sym_map_key`` on load."""
    gen = matcher.resolve(record)
    _augment_record_budget_params(record, gen)
    sym_int: Dict[str, int] = {}
    for name, value in record.params.items():
        if value is None or isinstance(value, bool):
            continue
        if not include_budget and _is_budget_name(str(name)):
            continue
        try:
            sym_int[str(name)] = int(value)
        except (TypeError, ValueError):
            continue
    if not sym_int:
        return None
    return make_sym_map_key(sym_int)


def _build_lookup_for_file(
    measure_record_path: Path,
    *,
    registry: GeneratorRegistry,
    output_path: Path,
    include_budget: bool,
) -> int:
    """Build a single lookup JSONL for ``measure_record_path``. Returns the
    number of unique sym_map entries written."""
    task_index = _parse_task_index_from_filename(measure_record_path)
    if task_index is None:
        print(f"  [skip-file] no task_index prefix: {measure_record_path.name}")
        return 0

    try:
        records = load_json_samples(measure_record_path)
    except Exception as err:  # pylint: disable=broad-except
        print(
            f"  [skip-file] cannot load {measure_record_path.name}: "
            f"{type(err).__name__}: {err}"
        )
        return 0

    if not records:
        print(f"  [empty] {measure_record_path.name}")
        return 0

    matcher = _SketchMatcher(registry, task_index)
    workload_key = matcher.workload_key

    groups: Dict[SymKey, List[float]] = defaultdict(list)
    total = len(records)
    skipped_no_cost = 0
    skipped_no_key = 0
    for rec in records:
        cost = rec.cost
        if cost is None or not math.isfinite(float(cost)):
            skipped_no_cost += 1
            continue
        # ``rec.workload_key`` should match ``matcher.workload_key`` because
        # the input file is named after one task — guard against cross-task
        # contamination just in case.
        rec_wk = getattr(rec, "workload_key", None)
        if rec_wk and rec_wk != workload_key:
            skipped_no_key += 1
            continue
        try:
            sym_key = _compute_sym_key(rec, matcher, include_budget=include_budget)
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"  [skip-record] {rec.sample_id}: "
                f"{type(err).__name__}: {err}"
            )
            skipped_no_key += 1
            continue
        if sym_key is None:
            skipped_no_key += 1
            continue
        groups[sym_key].append(float(cost))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for sym_key, neg_log_costs in groups.items():
            if not neg_log_costs:
                continue
            # ``rec.cost`` lives in ``-log(raw)`` space (see
            # adapter._transform_cost_for_training). Averaging in log-space
            # then inverting gives the geometric mean of raw seconds — which
            # is the value train.py expects to find under ``cost`` in the
            # on-disk lookup.
            avg_neg_log = float(sum(neg_log_costs) / len(neg_log_costs))
            if not math.isfinite(avg_neg_log):
                continue
            raw_cost = cost_label_to_raw(avg_neg_log, "neg_log")
            if raw_cost is None or not math.isfinite(raw_cost):
                continue
            sym_map = {str(name): int(value) for name, value in sym_key}
            f.write(
                json.dumps(
                    {
                        "workload_key": workload_key,
                        "sym_map": sym_map,
                        "cost": raw_cost,
                        "n_measurements": len(neg_log_costs),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    tmp_path.replace(output_path)
    print(
        f"  task={task_index} records={total} usable={total-skipped_no_cost-skipped_no_key} "
        f"skipped_no_cost={skipped_no_cost} skipped_no_key={skipped_no_key} "
        f"unique={written} → {output_path.name}"
    )
    return written


def _iter_input_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of measure-record JSONs (default: {DEFAULT_INPUT_DIR}).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write lookup_sym_maps JSONL files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    ap.add_argument(
        "--network-info-folder",
        default=DEFAULT_NETWORK_INFO_FOLDER,
        help=(
            "network_info_all directory used to resolve task → generator "
            f"(default: {DEFAULT_NETWORK_INFO_FOLDER})."
        ),
    )
    ap.add_argument(
        "--include-budget",
        action="store_true",
        help=(
            "Include thread_budget / vthread_budget in the sym_key. Off by "
            "default to match walks run with config.data.budget=False."
        ),
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")

    files = _iter_input_files(input_dir)
    if not files:
        raise SystemExit(f"no *.json files in {input_dir}")

    print(f"input_dir:  {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"include_budget={args.include_budget}")
    print(f"files: {len(files)}")

    registry = GeneratorRegistry(args.network_info_folder)
    total_written = 0
    files_done = 0
    for src in files:
        out = output_dir / (src.stem + ".jsonl")
        try:
            n = _build_lookup_for_file(
                src,
                registry=registry,
                output_path=out,
                include_budget=bool(args.include_budget),
            )
        except Exception as err:  # pylint: disable=broad-except
            print(f"  [error] {src.name}: {type(err).__name__}: {err}")
            continue
        total_written += n
        if n:
            files_done += 1

    print(
        f"\ndone: files={files_done}/{len(files)} unique_sym_maps_total={total_written}"
    )


if __name__ == "__main__":
    main()
