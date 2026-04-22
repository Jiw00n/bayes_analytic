"""Print the raw vthread extent expressions for task 1490."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget import train as train_module


def main() -> None:
    cfg = build_config()
    cfg.data.task_index = 1490
    resolve_task_paths(cfg)
    cfg.generator.hw_param = {"max_vthread_extent": 15}
    cfg.generator.disable_constraint = []

    registry = train_module.GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=cfg.generator.hw_param,
        disable_constraint=cfg.generator.disable_constraint,
    )
    bundle = train_module.build_dataset_bundle(cfg, registry)
    sample = bundle.train_dataset[0]
    gen = registry.get_generator(
        workload_key=sample.workload_key,
        target_kind=sample.target_kind,
        sketch_index=int(sample.sketch_index),
    )

    print(f"[diag] task_index=1490 sketch={sample.sketch_index}")
    print(f"[diag] _vthread_clamped_sp_names = {sorted(gen._vthread_clamped_sp_names)}")
    print()

    extents = gen.s.get_vthread_extents()
    print(f"[diag] number of vthread-annotated iters = {len(extents)}")
    for i, (sid, iid, extent) in enumerate(extents):
        print(f"  [{i}] stage_id={sid} iter_id={iid}")
        print(f"      extent = {extent}")

    print()
    print("[diag] number of thread-annotated iters =",
          len(gen.s.get_thread_extents()))
    for i, (sid, iid, extent) in enumerate(gen.s.get_thread_extents()):
        print(f"  [{i}] stage_id={sid} iter_id={iid}  extent={extent}")


if __name__ == "__main__":
    main()
