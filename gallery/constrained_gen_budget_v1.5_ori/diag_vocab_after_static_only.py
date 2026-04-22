"""Quickly build the dataset bundle with the static-only patch applied and
print tokenizer vocab size. Disables candidate-mask precompute to keep it fast.
"""
import sys
from pathlib import Path

# Ensure the gallery dir is on sys.path so `latent_model_budget` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_model_budget.config import build_config
from latent_model_budget.adapter import GeneratorRegistry
from latent_model_budget.dataset import build_dataset_bundle
import latent_model_budget.dataset as ds_mod


def _run_once(label: str):
    cfg = build_config()
    cfg.train.precompute_candidate_masks = False
    cfg.data.task_index = 1490
    from latent_model_budget.config import resolve_task_paths
    resolve_task_paths(cfg)
    print(f"[diag:{label}] task_index={cfg.data.task_index} json_paths={len(cfg.data.json_paths)}")
    registry = GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=getattr(cfg.generator, "hw_param", None),
        disable_constraint=getattr(cfg.generator, "disable_constraint", None),
    )
    bundle = build_dataset_bundle(cfg, registry)
    tok = bundle.tokenizer
    print(f"[diag:{label}] FINAL vocab={len(tok.id_to_token)} vars={len(tok.id_to_var)}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "patched"

    if mode == "baseline":
        # Monkey-patch the collector to behave as the pre-patch version:
        # union ALL SP names (including vectorize-followed ones).
        _orig = ds_mod._collect_generator_domain_values

        def _no_filter_collector(gen, order, *, include_budget=True):
            return _orig(gen, order, include_budget=include_budget)

        # Temporarily strip gen._vectorize_split_step_indices so the filter
        # in build_dataset_bundle becomes a no-op.
        from modules import schedule_generator as sg

        _orig_init = sg.ScheduleGenerator.__init__

        def _init_no_vec(self, *a, **kw):
            _orig_init(self, *a, **kw)
            self._vectorize_split_step_indices = set()

        sg.ScheduleGenerator.__init__ = _init_no_vec
        _run_once("baseline_no_filter")
        sg.ScheduleGenerator.__init__ = _orig_init
    else:
        _run_once("static_only")


if __name__ == "__main__":
    main()
