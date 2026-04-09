from __future__ import annotations

"""
decoder_only_dynamic_side_features.py

LatentParamVAE의 encoder / cost 경로는 그대로 두고,
decoder 입력에만 prefix-dependent dynamic side feature를 추가하는 유틸.

핵심 의도
- encode() 경로는 전혀 수정하지 않는다.
- 따라서 z / memory / cost_head가 보는 latent geometry는 유지한다.
- decode() 안에서만 base embedding 위에 dynamic side embedding을 더한다.

권장 사용 순서
1) 원래 모델 생성
2) checkpoint load_state_dict()
3) install_decoder_only_dynamic_side_features(...)
4) finetune
"""

from types import MethodType
from typing import Any, Iterable, Sequence
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_INT_RE = re.compile(r"[-+]?\d+")


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


def _family_of(var_name: str) -> str:
    name = str(var_name)
    if name.startswith("sp_"):
        return "sp"
    if name.startswith("ur_"):
        return "ur"
    if name.startswith("thread_budget"):
        return "thread_budget"
    if name.startswith("vthread_budget"):
        return "vthread_budget"
    return "other"


_FAMILIES: tuple[str, ...] = ("sp", "ur", "thread_budget", "vthread_budget", "other")
_FAMILY_TO_ID = {name: idx for idx, name in enumerate(_FAMILIES)}


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
    if value is None or denom <= 0.0:
        return 0.0
    return float(value) / float(denom)


class _VarMeta:
    __slots__ = (
        "family_onehot",
        "family_id",
        "outer_norm",
        "inner_norm",
        "has_outer",
        "has_inner",
        "group_id",
        "max_group_id",
    )

    def __init__(
        self,
        *,
        family_onehot: torch.Tensor,
        family_id: torch.Tensor,
        outer_norm: torch.Tensor,
        inner_norm: torch.Tensor,
        has_outer: torch.Tensor,
        has_inner: torch.Tensor,
        group_id: torch.Tensor,
        max_group_id: int,
    ) -> None:
        self.family_onehot = family_onehot
        self.family_id = family_id
        self.outer_norm = outer_norm
        self.inner_norm = inner_norm
        self.has_outer = has_outer
        self.has_inner = has_inner
        self.group_id = group_id
        self.max_group_id = int(max_group_id)


class _TokenMeta:
    __slots__ = (
        "has_int",
        "signed_norm",
        "log_norm",
        "is_zero",
        "is_one",
        "is_pow2",
    )

    def __init__(
        self,
        *,
        has_int: torch.Tensor,
        signed_norm: torch.Tensor,
        log_norm: torch.Tensor,
        is_zero: torch.Tensor,
        is_one: torch.Tensor,
        is_pow2: torch.Tensor,
    ) -> None:
        self.has_int = has_int
        self.signed_norm = signed_norm
        self.log_norm = log_norm
        self.is_zero = is_zero
        self.is_one = is_one
        self.is_pow2 = is_pow2


def _build_token_meta(id_to_token: Iterable[Any]) -> _TokenMeta:
    tokens = list(id_to_token)
    parsed = [_try_parse_int(tok) for tok in tokens]
    abs_vals = [abs(v) for v in parsed if v is not None]
    max_abs = max(abs_vals) if abs_vals else 1
    max_log = max(math.log1p(max_abs), 1e-6)

    has_int = []
    signed_norm = []
    log_norm = []
    is_zero = []
    is_one = []
    is_pow2 = []

    for v in parsed:
        vv = 0 if v is None else int(v)
        has_int.append(1.0 if v is not None else 0.0)
        signed_norm.append(_normalize_signed(v if v is not None else None, max_abs))
        log_norm.append(math.log1p(abs(vv)) / max_log if v is not None else 0.0)
        is_zero.append(1.0 if (v is not None and vv == 0) else 0.0)
        is_one.append(1.0 if (v is not None and vv == 1) else 0.0)
        is_pow2.append(_is_power_of_two(vv))

    return _TokenMeta(
        has_int=torch.tensor(has_int, dtype=torch.float32),
        signed_norm=torch.tensor(signed_norm, dtype=torch.float32),
        log_norm=torch.tensor(log_norm, dtype=torch.float32),
        is_zero=torch.tensor(is_zero, dtype=torch.float32),
        is_one=torch.tensor(is_one, dtype=torch.float32),
        is_pow2=torch.tensor(is_pow2, dtype=torch.float32),
    )


def _build_var_meta(id_to_var: Iterable[Any]) -> _VarMeta:
    vars_ = list(id_to_var)
    parsed = [_extract_var_indices(v) for v in vars_]
    max_outer = max([max(o, 0) for o, _ in parsed], default=1)
    max_inner = max([max(i, 0) for _, i in parsed], default=1)

    family_ids = []
    family_onehots = []
    outer_norm = []
    inner_norm = []
    has_outer = []
    has_inner = []
    group_id = []

    next_group_id = 1
    outer_to_group: dict[int, int] = {}

    for var_name, (outer_idx, inner_idx) in zip(vars_, parsed):
        fam_name = _family_of(str(var_name))
        fam_id = _FAMILY_TO_ID[fam_name]
        fam_onehot = [0.0] * len(_FAMILIES)
        fam_onehot[fam_id] = 1.0

        family_ids.append(fam_id)
        family_onehots.append(fam_onehot)
        outer_norm.append(_normalize_signed(outer_idx if outer_idx >= 0 else None, max_outer))
        inner_norm.append(_normalize_signed(inner_idx if inner_idx >= 0 else None, max_inner))
        has_outer.append(1.0 if outer_idx >= 0 else 0.0)
        has_inner.append(1.0 if inner_idx >= 0 else 0.0)

        if outer_idx >= 0:
            if outer_idx not in outer_to_group:
                outer_to_group[outer_idx] = next_group_id
                next_group_id += 1
            group_id.append(outer_to_group[outer_idx])
        else:
            group_id.append(0)

    return _VarMeta(
        family_onehot=torch.tensor(family_onehots, dtype=torch.float32),
        family_id=torch.tensor(family_ids, dtype=torch.long),
        outer_norm=torch.tensor(outer_norm, dtype=torch.float32),
        inner_norm=torch.tensor(inner_norm, dtype=torch.float32),
        has_outer=torch.tensor(has_outer, dtype=torch.float32),
        has_inner=torch.tensor(has_inner, dtype=torch.float32),
        group_id=torch.tensor(group_id, dtype=torch.long),
        max_group_id=max(group_id) if group_id else 0,
    )


def _collect_special_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for attr_name in (
        "pad_id",
        "bos_id",
        "eos_id",
        "unk_id",
        "mask_id",
        "cls_id",
        "sep_id",
    ):
        value = getattr(tokenizer, attr_name, None)
        if isinstance(value, int) and value >= 0:
            ids.append(int(value))
    return sorted(set(ids))


class DecoderDynamicTokenSideEmbedding(nn.Module):
    def __init__(
        self,
        *,
        token_meta: _TokenMeta,
        var_meta: _VarMeta,
        special_token_ids: Sequence[int],
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("token_has_int", token_meta.has_int, persistent=True)
        self.register_buffer("token_signed_norm", token_meta.signed_norm, persistent=True)
        self.register_buffer("token_log_norm", token_meta.log_norm, persistent=True)
        self.register_buffer("token_is_zero", token_meta.is_zero, persistent=True)
        self.register_buffer("token_is_one", token_meta.is_one, persistent=True)
        self.register_buffer("token_is_pow2", token_meta.is_pow2, persistent=True)

        self.register_buffer("var_family_onehot", var_meta.family_onehot, persistent=True)
        self.register_buffer("var_family_id", var_meta.family_id, persistent=True)
        self.register_buffer("var_outer_norm", var_meta.outer_norm, persistent=True)
        self.register_buffer("var_inner_norm", var_meta.inner_norm, persistent=True)
        self.register_buffer("var_has_outer", var_meta.has_outer, persistent=True)
        self.register_buffer("var_has_inner", var_meta.has_inner, persistent=True)
        self.register_buffer("var_group_id", var_meta.group_id, persistent=True)

        self.num_families = int(var_meta.family_onehot.shape[1])
        self.num_groups = int(var_meta.max_group_id + 1)
        self.special_token_ids = tuple(int(x) for x in special_token_ids)

        # static token(6) + static var(F+4) + dynamic(11)
        in_dim = 6 + self.num_families + 4 + 11
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _valid_prefix_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        valid = self.token_has_int[token_ids] > 0.0
        for token_id in self.special_token_ids:
            valid = valid & token_ids.ne(int(token_id))
        return valid

    def forward(self, token_ids: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        dtype = self.token_log_norm.dtype

        has_int = self.token_has_int[token_ids]
        token_signed = self.token_signed_norm[token_ids]
        token_log = self.token_log_norm[token_ids]
        token_is_zero = self.token_is_zero[token_ids]
        token_is_one = self.token_is_one[token_ids]
        token_is_pow2 = self.token_is_pow2[token_ids]

        family_oh = self.var_family_onehot[var_ids]
        family_id = self.var_family_id[var_ids]
        outer_norm = self.var_outer_norm[var_ids]
        inner_norm = self.var_inner_norm[var_ids]
        has_outer = self.var_has_outer[var_ids]
        has_inner = self.var_has_inner[var_ids]
        group_id = self.var_group_id[var_ids]

        valid_mask = self._valid_prefix_mask(token_ids)
        validf = valid_mask.to(dtype=dtype)

        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(bsz, seq_len)
        pos_norm = pos / max(seq_len - 1, 1)

        token_log_valid = token_log * validf
        prev_count_raw = torch.cumsum(validf, dim=1) - validf
        prev_logsum_all_raw = torch.cumsum(token_log_valid, dim=1) - token_log_valid
        prev_count_norm = prev_count_raw / max(seq_len - 1, 1)
        prev_logsum_all_norm = prev_logsum_all_raw / max(seq_len - 1, 1)

        prev_token_log = F.pad(token_log[:, :-1], (1, 0), value=0.0)
        prev_token_signed = F.pad(token_signed[:, :-1], (1, 0), value=0.0)
        prev_token_has_int = F.pad(has_int[:, :-1], (1, 0), value=0.0)

        fam_valid = family_oh * validf.unsqueeze(-1)
        fam_log_valid = family_oh * token_log_valid.unsqueeze(-1)
        fam_prev_count_all = torch.cumsum(fam_valid, dim=1) - fam_valid
        fam_prev_log_all = torch.cumsum(fam_log_valid, dim=1) - fam_log_valid
        fam_gather_index = family_id.unsqueeze(-1)
        prev_same_family_count_raw = torch.gather(fam_prev_count_all, dim=-1, index=fam_gather_index).squeeze(-1)
        prev_same_family_log_raw = torch.gather(fam_prev_log_all, dim=-1, index=fam_gather_index).squeeze(-1)
        prev_same_family_count_norm = prev_same_family_count_raw / max(seq_len - 1, 1)
        prev_same_family_log_norm = prev_same_family_log_raw / max(seq_len - 1, 1)
        prev_same_family_ratio = prev_same_family_count_raw / prev_count_raw.clamp_min(1.0)

        if self.num_groups <= 1:
            prev_same_group_count_raw = torch.zeros_like(prev_count_raw)
            prev_same_group_log_raw = torch.zeros_like(prev_logsum_all_raw)
        else:
            group_oh = F.one_hot(group_id.clamp_min(0), num_classes=self.num_groups).to(dtype=dtype)
            valid_group = (group_id > 0).to(dtype=dtype)
            group_valid = group_oh * (validf * valid_group).unsqueeze(-1)
            group_log_valid = group_oh * (token_log_valid * valid_group).unsqueeze(-1)
            group_prev_count_all = torch.cumsum(group_valid, dim=1) - group_valid
            group_prev_log_all = torch.cumsum(group_log_valid, dim=1) - group_log_valid
            group_gather_index = group_id.unsqueeze(-1)
            prev_same_group_count_raw = torch.gather(group_prev_count_all, dim=-1, index=group_gather_index).squeeze(-1)
            prev_same_group_log_raw = torch.gather(group_prev_log_all, dim=-1, index=group_gather_index).squeeze(-1)
            prev_same_group_count_raw = torch.where(group_id > 0, prev_same_group_count_raw, torch.zeros_like(prev_same_group_count_raw))
            prev_same_group_log_raw = torch.where(group_id > 0, prev_same_group_log_raw, torch.zeros_like(prev_same_group_log_raw))

        prev_same_group_count_norm = prev_same_group_count_raw / max(seq_len - 1, 1)
        prev_same_group_log_norm = prev_same_group_log_raw / max(seq_len - 1, 1)
        prev_same_group_ratio = prev_same_group_count_raw / prev_count_raw.clamp_min(1.0)

        static_token = torch.stack(
            [has_int, token_signed, token_log, token_is_zero, token_is_one, token_is_pow2],
            dim=-1,
        )
        static_var = torch.cat(
            [
                family_oh,
                outer_norm.unsqueeze(-1),
                inner_norm.unsqueeze(-1),
                has_outer.unsqueeze(-1),
                has_inner.unsqueeze(-1),
            ],
            dim=-1,
        )
        dynamic_feat = torch.stack(
            [
                pos_norm,
                prev_count_norm,
                prev_logsum_all_norm,
                prev_same_family_count_norm,
                prev_same_family_log_norm,
                prev_same_family_ratio,
                prev_same_group_count_norm,
                prev_same_group_log_norm,
                prev_same_group_ratio,
                prev_token_log,
                prev_token_signed + prev_token_has_int * 0.0,
            ],
            dim=-1,
        )

        feat = torch.cat([static_token, static_var, dynamic_feat], dim=-1)
        return self.scale * self.mlp(feat)


def build_decoder_only_dynamic_side_module(
    tokenizer: Any,
    *,
    d_model: int,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    scale_init: float = 0.10,
) -> DecoderDynamicTokenSideEmbedding:
    token_meta = _build_token_meta(tokenizer.id_to_token)
    var_meta = _build_var_meta(tokenizer.id_to_var)
    special_token_ids = _collect_special_token_ids(tokenizer)
    return DecoderDynamicTokenSideEmbedding(
        token_meta=token_meta,
        var_meta=var_meta,
        special_token_ids=special_token_ids,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        scale_init=scale_init,
    )


def install_decoder_only_dynamic_side_features(
    model: nn.Module,
    tokenizer: Any,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    scale_init: float = 0.10,
    module_name: str = "decoder_dynamic_side_emb",
) -> nn.Module:
    """
    encoder / cost 경로는 그대로 두고, decoder 입력에만 dynamic side feature를 붙인다.

    사용 예시
    --------
    model = LatentParamVAE(...)
    model.load_state_dict(payload["model_state"])
    install_decoder_only_dynamic_side_features(model, tokenizer)
    """
    if not hasattr(model, "cfg") or not hasattr(model.cfg, "d_model"):
        raise AttributeError("model.cfg.d_model is required")
    if hasattr(model, module_name):
        return model

    side_module = build_decoder_only_dynamic_side_module(
        tokenizer,
        d_model=int(model.cfg.d_model),
        hidden_dim=hidden_dim,
        dropout=dropout,
        scale_init=scale_init,
    )
    device = next(model.parameters()).device
    setattr(model, module_name, side_module.to(device))

    original_decode = model.decode

    def _decode_with_decoder_only_dynamic(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_var_ids: torch.Tensor,
        memory: torch.Tensor,
        latent_cond: torch.Tensor,
        decoder_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # base embedding은 기존 경로 그대로 사용한다.
        y = self._embed(decoder_input_ids, decoder_var_ids)
        y = y + getattr(self, module_name)(decoder_input_ids, decoder_var_ids)
        seq_len = y.size(1)
        causal_mask = self._causal_mask(seq_len, y.device)
        dec = self.decoder(
            tgt=y,
            memory=memory,
            latent_cond=latent_cond,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_pad_mask,
        )
        return self.lm_head(dec)

    # decode만 patch. encode / forward의 enc path / cost_head는 그대로 유지된다.
    model.decode = MethodType(_decode_with_decoder_only_dynamic, model)
    # 나중에 원복하고 싶을 때 참고용으로 저장.
    model._original_decode_without_decoder_dynamic = original_decode
    return model
