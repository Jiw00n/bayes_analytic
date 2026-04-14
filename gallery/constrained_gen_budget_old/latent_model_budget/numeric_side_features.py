from __future__ import annotations

"""
numeric_side_features.py

기존 LatentParamVAE에 "파라미터의 수치적 의미"를 side embedding으로
붙이기 위한 최소 침습(minimal-invasive) 유틸.

핵심 아이디어
- 기존 token_emb + var_emb + pos_emb 경로는 그대로 둔다.
- token string / var name에서 뽑은 numerical/static feature를 작은 MLP로 d_model에 투영한다.
- 그 결과를 _embed()에 더해준다.
"""

from types import MethodType
from typing import Any, Iterable
import math
import re

import torch
import torch.nn as nn

_INT_RE = re.compile(r"[-+]?\d+")


def _safe_float(x: float, default: float = 0.0) -> float:
    if x is None:
        return default
    if not math.isfinite(float(x)):
        return default
    return float(x)


def _try_parse_int(text: str) -> int | None:
    text = str(text).strip()
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        pass
    m = _INT_RE.search(text)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _is_power_of_two(v: int) -> float:
    if v <= 0:
        return 0.0
    return 1.0 if (v & (v - 1)) == 0 else 0.0


def _family_one_hot(var_name: str) -> list[float]:
    name = str(var_name)
    fams = ["sp_", "ur_", "thread_budget", "vthread_budget"]
    out = [0.0] * (len(fams) + 1)
    for i, prefix in enumerate(fams):
        if name.startswith(prefix):
            out[i] = 1.0
            return out
    out[-1] = 1.0
    return out


def _extract_var_indices(var_name: str) -> tuple[int, int]:
    name = str(var_name)
    if name.startswith("sp_"):
        parts = name.split("_")
        if len(parts) >= 3:
            try:
                return int(parts[1]), int(parts[2])
            except Exception:
                return -1, -1
    if name.startswith("ur_"):
        parts = name.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1]), -1
            except Exception:
                return -1, -1
    return -1, -1


def _normalize_signed(value: int | None, denom: float) -> float:
    if value is None or denom <= 0:
        return 0.0
    return float(value) / float(denom)


def _build_token_feature_table(id_to_token: Iterable[Any]) -> torch.Tensor:
    tokens = list(id_to_token)
    parsed = [_try_parse_int(tok) for tok in tokens]
    abs_vals = [abs(v) for v in parsed if v is not None]
    max_abs = max(abs_vals) if abs_vals else 1
    max_log = math.log1p(max_abs)

    rows: list[list[float]] = []
    for v in parsed:
        has_int = 1.0 if v is not None else 0.0
        vv = 0 if v is None else int(v)
        signed_norm = _normalize_signed(vv if v is not None else None, max_abs)
        log_norm = (
            math.log1p(abs(vv)) / max_log
            if (v is not None and max_log > 0.0)
            else 0.0
        )
        is_pos = 1.0 if (v is not None and vv > 0) else 0.0
        is_neg = 1.0 if (v is not None and vv < 0) else 0.0
        is_zero = 1.0 if (v is not None and vv == 0) else 0.0
        is_pow2 = _is_power_of_two(vv)
        recip = 0.0
        if v is not None and vv != 0:
            recip = max(min(1.0 / float(vv), 4.0), -4.0) / 4.0
        parity_even = 1.0 if (v is not None and vv % 2 == 0) else 0.0
        parity_odd = 1.0 if (v is not None and vv % 2 != 0) else 0.0
        rows.append([
            has_int,
            _safe_float(signed_norm),
            _safe_float(log_norm),
            is_pos,
            is_neg,
            is_zero,
            is_pow2,
            _safe_float(recip),
            parity_even,
            parity_odd,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def _build_var_feature_table(id_to_var: Iterable[Any]) -> torch.Tensor:
    vars_ = list(id_to_var)
    parsed = [_extract_var_indices(v) for v in vars_]
    max_outer = max([max(o, 0) for o, _ in parsed], default=1)
    max_inner = max([max(i, 0) for _, i in parsed], default=1)

    rows: list[list[float]] = []
    for var_name, (outer_idx, inner_idx) in zip(vars_, parsed):
        fam = _family_one_hot(str(var_name))
        has_inner = 1.0 if inner_idx >= 0 else 0.0
        rows.append(
            fam
            + [
                _normalize_signed(outer_idx if outer_idx >= 0 else None, max_outer),
                _normalize_signed(inner_idx if inner_idx >= 0 else None, max_inner),
                has_inner,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


class NumericTokenSideEmbedding(nn.Module):
    def __init__(
        self,
        *,
        token_feature_table: torch.Tensor,
        var_feature_table: torch.Tensor,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("token_feature_table", token_feature_table, persistent=True)
        self.register_buffer("var_feature_table", var_feature_table, persistent=True)
        in_dim = int(token_feature_table.size(1) + var_feature_table.size(1))
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, token_ids: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        token_feat = self.token_feature_table[token_ids]
        var_feat = self.var_feature_table[var_ids]
        feat = torch.cat([token_feat, var_feat], dim=-1)
        return self.scale * self.mlp(feat)


def build_numeric_side_module(
    tokenizer,
    *,
    d_model: int,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    scale_init: float = 0.10,
) -> NumericTokenSideEmbedding:
    token_table = _build_token_feature_table(tokenizer.id_to_token)
    var_table = _build_var_feature_table(tokenizer.id_to_var)
    return NumericTokenSideEmbedding(
        token_feature_table=token_table,
        var_feature_table=var_table,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        scale_init=scale_init,
    )


def install_numeric_side_features(
    model: nn.Module,
    tokenizer,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    scale_init: float = 0.10,
    module_name: str = "numeric_side_emb",
) -> nn.Module:
    if not hasattr(model, "cfg") or not hasattr(model.cfg, "d_model"):
        raise AttributeError("model.cfg.d_model is required")
    if hasattr(model, module_name):
        return model

    side_module = build_numeric_side_module(
        tokenizer,
        d_model=int(model.cfg.d_model),
        hidden_dim=hidden_dim,
        dropout=dropout,
        scale_init=scale_init,
    )
    setattr(model, module_name, side_module.to(next(model.parameters()).device))

    def _embed_with_numeric(self, token_ids: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        pos = self._positions(bsz, seq_len, token_ids.device)
        x = self.token_emb(token_ids) + self.var_emb(var_ids) + self.pos_emb(pos)
        x = x + getattr(self, module_name)(token_ids, var_ids)
        return self.dropout(x)

    model._embed = MethodType(_embed_with_numeric, model)
    return model
