from __future__ import annotations

import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget.config import build_config
from latent_model_budget.adapter import (
    GeneratorRegistry,
    _record_split_extent_overrides,
    load_json_samples,
)
from latent_model_budget.dataset import (
    _build_prepared_sample,
    build_dataset_bundle,
    get_model_param_order,
)
from modules.sym_types import eval_sym_extent


def _find_measure_record(records, suffix: str):
    for record in records:
        if record.sample_id.endswith(suffix):
            return record
    raise AssertionError(f"measure record with suffix={suffix} not found")


def _raw_step_payload(record, step_idx: int):
    steps = record.raw["i"][1][1]
    return steps[step_idx]


@pytest.fixture(scope="module")
def cfg():
    config = build_config()
    config.data.json_paths = [
        "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json"
    ]
    config.train.precompute_candidate_masks = False
    config.data.filter_invalid_records = False
    return config


@pytest.fixture(scope="module")
def records(cfg):
    return load_json_samples(cfg.data.json_paths[0])


@pytest.fixture(scope="module")
def registry(cfg):
    return GeneratorRegistry(cfg.data.network_info_folder)


@pytest.fixture(scope="module")
def bundle(cfg, registry):
    return build_dataset_bundle(cfg, registry)


def test_generic_replay_recovers_step31_split_extent(records, registry):
    record = _find_measure_record(records, "::r1823")
    step31 = _raw_step_payload(record, 31)
    assert step31[:4] == ["SP", 2, 0, 28]

    gen = registry.get_generator_from_record(record)
    overrides = _record_split_extent_overrides(record)
    assert gen._get_dynamic_split_extent(
        31, sym_map=record.params, extent_overrides=overrides
    ) == 28

    oracle = registry.build_oracle_from_record(record)
    order = get_model_param_order(gen, include_budget=False)
    values = [int(record.params[name]) for name in order]

    for name, value in zip(order, values):
        if name == "sp_31_0":
            break
        oracle.assign(name, value)

    assert gen._get_dynamic_split_extent(
        31, sym_map=oracle._sym_map, extent_overrides=overrides
    ) == 28
    assert max(oracle.candidate_values("sp_31_0")) <= 28


def test_dataset_tokenizer_covers_observed_gold_values(bundle):
    tokenizer = bundle.tokenizer
    all_samples = (
        list(bundle.train_dataset.samples)
        + list(bundle.val_dataset.samples)
        + list(bundle.test_dataset.samples)
    )

    assert "64" in tokenizer.token_to_id
    for sample in all_samples:
        for name, value in zip(sample.ordered_param_names, sample.ordered_param_values):
            token = tokenizer.value_to_token(name, int(value))
            assert token in tokenizer.token_to_id, (sample.sample_id, name, value)


def test_sampled_oracle_candidates_map_to_tokenizer(bundle, registry, cfg):
    tokenizer = bundle.tokenizer
    sample_records = list(bundle.train_records[:10]) + list(bundle.val_records[:10])

    for record in sample_records:
        oracle = registry.build_oracle_from_record(record)
        order = get_model_param_order(oracle.generator, include_budget=cfg.data.budget)
        values = [int(record.params[name]) for name in order]
        for name, value in zip(order, values):
            candidates = oracle.candidate_values(name)
            missing = [
                int(candidate)
                for candidate in candidates
                if tokenizer.value_to_token(name, int(candidate)) not in tokenizer.token_to_id
            ]
            assert not missing, (record.sample_id, name, missing)
            oracle.assign(name, value)


def test_candidate_mask_does_not_fallback_to_gold_singleton(records, registry, bundle):
    record = _find_measure_record(records, "::r63")
    gen = registry.get_generator_from_record(record)
    order = get_model_param_order(gen, include_budget=False)
    values = [int(record.params[name]) for name in order]
    sample = _build_prepared_sample(
        record,
        order,
        values,
        bundle.tokenizer,
        registry=registry,
        include_candidate_masks=True,
    )

    step_idx = order.index("sp_2_0")
    gold_token = bundle.tokenizer.value_to_token("sp_2_0", int(record.params["sp_2_0"]))
    gold_id = bundle.tokenizer.token_to_id[gold_token]

    assert sample.candidate_masks is not None
    assert bool(sample.candidate_masks[step_idx, gold_id])
    assert int(sample.candidate_masks[step_idx].sum().item()) > 1
