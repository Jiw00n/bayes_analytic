"""Pre-seed the per-task measurement lookup table from existing measurement
JSON files.

Given a ``task_index``, this script scans ``gallery/`` recursively for files
matching ``{task_index}_*.json``, reconstructs each record's full ``sym_map``
(matching the shape produced by ``build_walk_records`` at walk time), and
writes one JSONL entry per unique sym_map to
``checkpoints_all/{task_index}/{task_index}_measurement_lookup.jsonl``.

Duplicate sym_maps get their per-record ``mean_cost`` averaged.

Usage:
    python build_measurement_lookup.py <task_index>
"""

from __future__ import annotations

import argparse
import json
import math
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
    load_json_samples,
)
from latent_model_budget.config import DEFAULT_NETWORK_INFO_FOLDER  # noqa: E402
from modules.task_paths import clean_name  # noqa: E402
from result_csv_utils import make_sym_map_key  # noqa: E402


SymKey = Tuple[Tuple[str, int], ...]


def _task_file_core(task) -> str:
    """The substring every JSON filename for this task contains.

    Produced by ``clean_name((task.workload_key, str(task.target.kind)))`` —
    the same convention used by ``get_measure_record_filename`` and the
    ``{task_index}_<core>.json`` / ``{task_index}_ansor_<core>.json`` variants.
    Matching on this substring picks up all of those forms and excludes files
    that share only the hash but differ in shape (e.g. batch=2 vs batch=1)."""
    return clean_name((task.workload_key, str(task.target.kind)))


def _find_task_jsons(gallery_root: Path, task_core: str) -> List[Path]:
    """Every ``*.json`` file under ``gallery_root`` whose filename contains
    ``task_core``, sorted and de-duplicated by resolved path."""
    paths = set()
    for p in gallery_root.rglob("*.json"):
        if not p.is_file():
            continue
        if task_core in p.name:
            paths.add(p.resolve())
    return sorted(paths)


class _SketchMatcher:
    """Resolves ``record.param_signature`` → cached sketch-based generator.

    Rebuilding ``ScheduleGenerator.from_task_state`` per record (what
    ``GeneratorRegistry.get_generator_from_record`` does for measure-record
    payloads) is the slow path. This class routes every record through the
    sketch-index cache instead so only one generator is built per unique
    sketch signature."""

    def __init__(self, registry: GeneratorRegistry, task_index: int):
        self.registry = registry
        self.task_index = int(task_index)
        self.task = registry._resolve_task(task_index=self.task_index)
        self.workload_key, self.target_kind = registry._task_signature(self.task)
        self._sig_to_sketch_index: Dict[Tuple[str, ...], int] = {}
        self._generators: Dict[int, object] = {}
        self._sketches = registry._get_sketches_for_task(self.task)
        # Pre-index every sketch by its normalized (no-budget) param signature.
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
        """Return the cached generator matching ``record.param_signature``.
        Raises if no sketch matches."""
        if not record.param_signature:
            raise ValueError(f"record {record.sample_id} has no param_signature")
        normalized = _normalize_generator_signature(record.param_signature)
        sketch_index = self._sig_to_sketch_index.get(normalized)
        if sketch_index is None:
            # Fall back to a linear scan in case the pre-index missed a permutation.
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
    """Build the sym_key to match what ``build_walk_records`` produces.

    The walk builds its key from ``decoded.sym_map``, which is
    ``dict(oracle.generator.s.sym_map)`` overlaid with ``decoded_params`` (one
    entry per ``ordered_names`` returned by
    ``get_model_param_order(gen, include_budget=budget_enabled(config))``).
    When ``config.data.budget=False`` (the default), ``ordered_names`` excludes
    ``thread_budget`` / ``vthread_budget`` and ``generator.s.sym_map`` has never
    been populated with them either — so the walk's sym_key is sp_/ur_ only.
    We mirror that here: budget entries are dropped unless ``include_budget``
    is set."""
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


def _iter_records(paths: Iterable[Path]) -> Iterable[JsonSampleRecord]:
    for path in paths:
        try:
            samples = load_json_samples(path)
        except Exception as err:  # pylint: disable=broad-except
            print(f"  [skip-file] {path.name}: {type(err).__name__}: {err}")
            continue
        print(f"  loaded {len(samples)} records from {path.name}")
        for rec in samples:
            yield rec


def build_lookup(
    task_index: int,
    *,
    gallery_root: Path,
    network_info_folder: str,
    output_dir: Path,
    include_budget: bool,
) -> Path:
    registry = GeneratorRegistry(network_info_folder)
    matcher = _SketchMatcher(registry, task_index)
    task_core = _task_file_core(matcher.task)
    print(
        f"Task {task_index}: {len(matcher._sketches)} sketches cached "
        f"({matcher.workload_key}, {matcher.target_kind})"
    )
    print(f"Matching filename substring: {task_core}")
    print(f"include_budget={include_budget}")

    paths = _find_task_jsons(gallery_root, task_core)
    if not paths:
        raise SystemExit(
            f"No JSON files containing {task_core!r} found under {gallery_root}"
        )
    print(f"Found {len(paths)} JSON files for task {task_index}:")
    for p in paths:
        print(f"  {p.relative_to(gallery_root)}")

    groups: Dict[SymKey, List[float]] = defaultdict(list)
    total_records = 0
    usable_records = 0
    skipped_no_cost = 0
    skipped_no_key = 0

    for rec in _iter_records(paths):
        total_records += 1
        cost = rec.cost
        if cost is None or not math.isfinite(float(cost)):
            skipped_no_cost += 1
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
        usable_records += 1

    print(
        f"\nprocessed={total_records} usable={usable_records} "
        f"skipped_no_cost={skipped_no_cost} skipped_no_key={skipped_no_key}"
    )
    print(f"unique sym_maps: {len(groups)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{int(task_index)}_measurement_lookup.jsonl"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    written = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for sym_key, costs in groups.items():
            if not costs:
                continue
            # ``rec.cost`` is already ``-log(raw)`` (see
            # adapter._transform_cost_for_training); averaging in log-space then
            # ``exp(-avg)`` gives the geometric mean of raw costs, which is the
            # raw-cost value we persist to the lookup.
            avg_neg_log_cost = float(sum(costs) / len(costs))
            if not math.isfinite(avg_neg_log_cost):
                continue
            raw_cost = math.exp(-avg_neg_log_cost)
            if not math.isfinite(raw_cost):
                continue
            sym_map = {str(name): int(value) for name, value in sym_key}
            f.write(
                json.dumps(
                    {
                        "sym_map": sym_map,
                        "cost": raw_cost,
                        "n_measurements": len(costs),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    tmp_path.replace(output_path)
    print(f"wrote {written} entries → {output_path}")
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_index", type=int, help="Task index to aggregate.")
    ap.add_argument(
        "--gallery-root",
        default=str(GALLERY_ROOT),
        help=f"Directory to scan recursively (default: {GALLERY_ROOT}).",
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
        "--output-dir",
        default=None,
        help=(
            "Target directory for the lookup file. "
            "Defaults to checkpoints_all/<task_index>."
        ),
    )
    ap.add_argument(
        "--include-budget",
        action="store_true",
        help=(
            "Include thread_budget / vthread_budget in the sym_key. Off by "
            "default to match walks run with config.data.budget=False (the "
            "current default)."
        ),
    )
    args = ap.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else HERE / "checkpoints_all" / str(args.task_index)
    )
    build_lookup(
        task_index=int(args.task_index),
        gallery_root=Path(args.gallery_root),
        network_info_folder=args.network_info_folder,
        output_dir=output_dir,
        include_budget=bool(args.include_budget),
    )


if __name__ == "__main__":
    main()
