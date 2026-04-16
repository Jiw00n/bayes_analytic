"""Compare raw-divisor vs propagation-aware domains for task 1490."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tvm.auto_scheduler.measure_record import load_record_from_string
from latent_model_budget.adapter import GeneratorRegistry, _load_measure_record_sample
from latent_model_budget.dataset import (
    _build_prepared_sample,
    _collect_generator_domain_values,
    _collect_record_step_candidates,
    _get_generator_for_record,
    get_model_param_order,
)
from latent_model_budget.tokenizer import ParamTokenizer

RECORD_JSON = Path(
    "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/"
    "1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json"
)
NETWORK_INFO = "/root/work/tvm-ansor/gallery/dataset/network_info_all"


def _load_first_record():
    text = RECORD_JSON.read_text(encoding="utf-8")
    line = next(s for s in text.splitlines() if s.strip())
    return _load_measure_record_sample(RECORD_JSON, line, 0)


def main():
    registry = GeneratorRegistry(NETWORK_INFO)
    record = _load_first_record()
    gen = _get_generator_for_record(record, registry)
    order = get_model_param_order(gen, include_budget=True)

    raw = _collect_generator_domain_values(gen, order, include_budget=True)
    oracle = registry.build_oracle_from_record(record)
    step_cands, valid = _collect_record_step_candidates(record, order, oracle)

    print(f"order length: {len(order)}, valid={valid}")
    print(f"\n{'name':<22} {'raw':>4} {'prop':>4}  diff")
    raw_total = 0
    prop_total = 0
    for name, cands in zip(order, step_cands):
        raw_vals = sorted(raw.get(name, []))
        prop_vals = sorted({int(v) for v in cands})
        added = sorted(set(prop_vals) - set(raw_vals))
        removed = sorted(set(raw_vals) - set(prop_vals))
        raw_total += len(raw_vals)
        prop_total += len(prop_vals)
        marker = ""
        if added or removed:
            marker = f" +{added} -{removed}"
        print(f"{name:<22} {len(raw_vals):>4} {len(prop_vals):>4}{marker}")
    print(f"\nTOTAL raw={raw_total}  propagation-aware={prop_total}")

    # Build a tokenizer using BOTH old and new vocabs and compare sizes.
    raw_domain = {n: sorted(raw.get(n, [])) for n in order}
    prop_domain = {n: sorted({int(v) for v in c}) for n, c in zip(order, step_cands)}
    values = [int(record.params[n]) for n in order]
    tok_raw = ParamTokenizer.build(
        train_ordered_names=[order],
        train_ordered_values=[values],
        all_ordered_names=[order],
        domain_values_by_name=raw_domain,
    )
    tok_prop = ParamTokenizer.build(
        train_ordered_names=[order],
        train_ordered_values=[values],
        all_ordered_names=[order],
        domain_values_by_name=prop_domain,
    )
    print(
        f"\ntokenizer vocab: raw={len(tok_raw.id_to_token)}  prop={len(tok_prop.id_to_token)}"
    )

    sample = _build_prepared_sample(
        record,
        order,
        values,
        tok_prop,
        include_candidate_masks=True,
        precomputed_candidates=step_cands,
    )
    masks = sample.candidate_masks
    print(f"mask shape: {tuple(masks.shape)}  positive bits per row:")
    for name, row in zip(order, masks):
        print(f"  {name:<22} {int(row.sum().item())}")


if __name__ == "__main__":
    main()
