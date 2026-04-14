from __future__ import annotations

"""
constraint_state_features.py

정적 token/var 메타데이터 버퍼를 잔뜩 넣는 대신,
바깥(generator / constraint engine / training loop)에서 계산한
"작은 decision-state feature"를 decoder에만 주입하는 유틸.

핵심 원칙
- encoder / z / cost_head는 건드리지 않는다.
- decoder 쪽에만 state feature를 더한다.
- state feature의 의미는 모델 안이 아니라 바깥에서 만든다.
- feature 개수는 작게 유지한다. (권장 6~8개)

권장 feature 예시
1) num_valid_candidates_log
2) gold_rank_norm              (학습 시에만 가능)
3) chosen_value_log            (teacher forcing면 gold, AR면 chosen)
4) remaining_budget_norm
5) used_budget_norm
6) slack_norm
7) same_group_product_log
8) same_family_count_norm

권장 사용 순서
1) checkpoint load_state_dict()
2) install_decoder_only_constraint_state_features(model, feature_dim=8)
3) train / greedy decode 루프에서 decoder_state_features=[B,T,F] 계산
4) model(..., decoder_state_features=state_feats)
"""

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence
import math

import torch
import torch.nn as nn


FEATURE_NAMES: tuple[str, ...] = (
    "num_valid_candidates_log",
    "gold_rank_norm",
    "chosen_value_log",
    "remaining_budget_norm",
    "used_budget_norm",
    "slack_norm",
    "same_group_product_log",
    "same_family_count_norm",
)


class DecoderStateProjector(nn.Module):
    """Small MLP that projects external state features -> d_model."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(
                f"decoder_state_features must have shape [B, T, F], got {tuple(feats.shape)}"
            )
        if feats.size(-1) != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.feature_dim}, got {feats.size(-1)}"
            )
        return self.net(feats) * self.scale


class StepStateAdapter:
    """
    느슨한 dict 기반 adapter.

    generator state가 정확히 어떤 필드를 주는지 아직 고정하지 않았을 때,
    아래 키들 중 가능한 것만 사용해서 8차원 feature vector를 만든다.
    없는 값은 0으로 둔다.

    지원 키 예시
    - valid_token_ids: Sequence[int]
    - num_valid_candidates: int
    - gold_rank: int
    - chosen_value: int/float
    - remaining_budget: int/float
    - total_budget: int/float
    - used_budget: int/float
    - slack: int/float
    - same_group_product: int/float
    - same_family_count: int/float
    """

    feature_names = FEATURE_NAMES

    @staticmethod
    def _f(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            value = float(value)
        except Exception:
            return float(default)
        if math.isnan(value) or math.isinf(value):
            return float(default)
        return float(value)

    @classmethod
    def from_step_info(cls, info: Mapping[str, Any]) -> list[float]:
        valid_token_ids = info.get("valid_token_ids")
        if valid_token_ids is not None:
            try:
                n_valid = len(valid_token_ids)
            except Exception:
                n_valid = None
        else:
            n_valid = None
        if n_valid is None:
            n_valid = info.get("num_valid_candidates")
        n_valid = max(int(cls._f(n_valid, 0.0)), 0)
        num_valid_candidates_log = math.log1p(float(n_valid))

        gold_rank = info.get("gold_rank")
        if gold_rank is None or n_valid <= 1:
            gold_rank_norm = 0.0
        else:
            gold_rank_norm = cls._f(gold_rank, 0.0) / float(max(n_valid - 1, 1))

        chosen_value = cls._f(info.get("chosen_value"), 0.0)
        chosen_value_log = math.copysign(math.log1p(abs(chosen_value)), chosen_value)

        total_budget = cls._f(info.get("total_budget"), 0.0)
        remaining_budget = cls._f(info.get("remaining_budget"), 0.0)
        used_budget = cls._f(info.get("used_budget"), 0.0)
        slack = cls._f(info.get("slack"), 0.0)
        denom = abs(total_budget) if abs(total_budget) > 1e-12 else 0.0

        remaining_budget_norm = remaining_budget / denom if denom > 0.0 else 0.0
        used_budget_norm = used_budget / denom if denom > 0.0 else 0.0
        slack_norm = slack / denom if denom > 0.0 else 0.0

        same_group_product = cls._f(info.get("same_group_product"), 0.0)
        same_group_product_log = math.copysign(math.log1p(abs(same_group_product)), same_group_product)

        same_family_count = cls._f(info.get("same_family_count"), 0.0)
        same_family_count_norm = min(max(same_family_count / 16.0, 0.0), 4.0)

        return [
            float(num_valid_candidates_log),
            float(gold_rank_norm),
            float(chosen_value_log),
            float(remaining_budget_norm),
            float(used_budget_norm),
            float(slack_norm),
            float(same_group_product_log),
            float(same_family_count_norm),
        ]

    @classmethod
    def batch_to_tensor(
        cls,
        batch_step_infos: Sequence[Sequence[Mapping[str, Any]]],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        pad_to_length: int | None = None,
    ) -> torch.Tensor:
        """
        batch_step_infos: length B, each item is length T_i list of dicts
        return: [B, T, F]
        """
        batch_size = len(batch_step_infos)
        max_len = pad_to_length if pad_to_length is not None else max((len(x) for x in batch_step_infos), default=0)
        feat_dim = len(cls.feature_names)
        out = torch.zeros(batch_size, max_len, feat_dim, dtype=dtype, device=device)
        for b, seq in enumerate(batch_step_infos):
            limit = min(len(seq), max_len)
            if limit <= 0:
                continue
            rows = [cls.from_step_info(seq[t]) for t in range(limit)]
            out[b, :limit] = torch.tensor(rows, dtype=dtype, device=device)
        return out


def install_decoder_only_constraint_state_features(
    model: nn.Module,
    *,
    feature_dim: int = len(FEATURE_NAMES),
    hidden_dim: int = 64,
    dropout: float = 0.0,
    scale_init: float = 0.10,
) -> nn.Module:
    """
    model.forward(..., decoder_state_features=[B,T,F]) 를 받을 수 있게 패치한다.
    encoder / cost 경로는 유지하고, decode()의 y에만 state projection을 더한다.
    """
    if hasattr(model, "decoder_state_projector"):
        return model

    d_model = int(getattr(model, "d_model"))
    projector = DecoderStateProjector(
        feature_dim=int(feature_dim),
        d_model=d_model,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        scale_init=float(scale_init),
    )
    projector.to(next(model.parameters()).device)
    model.add_module("decoder_state_projector", projector)
    model._decoder_state_features_ctx = None

    orig_forward = model.forward
    orig_decode = model.decode

    def _patched_forward(self, *args, **kwargs):
        decoder_state_features = kwargs.pop("decoder_state_features", None)
        prev = getattr(self, "_decoder_state_features_ctx", None)
        self._decoder_state_features_ctx = decoder_state_features
        try:
            return orig_forward(*args, **kwargs)
        finally:
            self._decoder_state_features_ctx = prev

    def _patched_decode(self, z, decoder_input_ids, decoder_var_ids, pad_token_id):
        # orig_decode 내부 구현을 복제하지 않기 위해, base y를 얻는 부분만 override하는 대신
        # 원래 decode 메서드에 들어가기 전에 _embed를 임시 패치한다.
        state_feats = getattr(self, "_decoder_state_features_ctx", None)
        if state_feats is None:
            return orig_decode(z, decoder_input_ids, decoder_var_ids, pad_token_id)

        base_embed = self._embed

        def _embed_with_state(token_ids, var_ids):
            y = base_embed(token_ids, var_ids)
            feats = state_feats
            if feats.device != y.device:
                feats_local = feats.to(device=y.device, dtype=y.dtype)
            else:
                feats_local = feats.to(dtype=y.dtype)
            if feats_local.shape[:2] != y.shape[:2]:
                raise ValueError(
                    f"decoder_state_features shape {tuple(feats_local.shape)} does not match decoder input shape {tuple(y.shape[:2])}"
                )
            return y + self.decoder_state_projector(feats_local)

        self._embed = _embed_with_state  # type: ignore[assignment]
        try:
            return orig_decode(z, decoder_input_ids, decoder_var_ids, pad_token_id)
        finally:
            self._embed = base_embed  # type: ignore[assignment]

    model.forward = MethodType(_patched_forward, model)
    model.decode = MethodType(_patched_decode, model)
    return model


@contextmanager
def decoder_state_features(model: nn.Module, feats: torch.Tensor | None):
    prev = getattr(model, "_decoder_state_features_ctx", None)
    model._decoder_state_features_ctx = feats
    try:
        yield model
    finally:
        model._decoder_state_features_ctx = prev


# ---------------------------------------------------------------------------
# Minimal helper for train / eval loops
# ---------------------------------------------------------------------------

def build_minimal_step_info(
    *,
    valid_token_ids: Sequence[int] | None = None,
    gold_rank: int | None = None,
    chosen_value: int | float | None = None,
    remaining_budget: int | float | None = None,
    total_budget: int | float | None = None,
    used_budget: int | float | None = None,
    slack: int | float | None = None,
    same_group_product: int | float | None = None,
    same_family_count: int | float | None = None,
) -> dict[str, Any]:
    return {
        "valid_token_ids": None if valid_token_ids is None else list(valid_token_ids),
        "gold_rank": gold_rank,
        "chosen_value": chosen_value,
        "remaining_budget": remaining_budget,
        "total_budget": total_budget,
        "used_budget": used_budget,
        "slack": slack,
        "same_group_product": same_group_product,
        "same_family_count": same_family_count,
    }


__all__ = [
    "FEATURE_NAMES",
    "DecoderStateProjector",
    "StepStateAdapter",
    "install_decoder_only_constraint_state_features",
    "decoder_state_features",
    "build_minimal_step_info",
]
