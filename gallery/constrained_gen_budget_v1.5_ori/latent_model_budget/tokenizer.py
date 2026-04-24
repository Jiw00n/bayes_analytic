from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


@dataclass
class ParamTokenizerState:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    var_to_id: Dict[str, int]
    id_to_var: List[str]


class ParamTokenizer:
    PAD = "<PAD>"
    BOS = "<BOS>"
    UNK = "<UNK>"

    def __init__(self, state: ParamTokenizerState):
        self.token_to_id = dict(state.token_to_id)
        self.id_to_token = list(state.id_to_token)
        self.var_to_id = dict(state.var_to_id)
        self.id_to_var = list(state.id_to_var)

        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.unk_id = self.token_to_id[self.UNK]
        self.var_pad_id = self.var_to_id["<VAR_PAD>"]

    @staticmethod
    def value_to_token(var_name: str, value: int) -> str:
        if var_name.startswith("ur_"):
            return f"auto_unroll_max_step${int(value)}"
        return str(int(value))

    @staticmethod
    def token_to_value(var_name: str, token: str) -> int | None:
        if token in {ParamTokenizer.PAD, ParamTokenizer.BOS, ParamTokenizer.UNK}:
            return None
        if var_name.startswith("ur_"):
            prefix = "auto_unroll_max_step$"
            if not token.startswith(prefix):
                return None
            return int(token[len(prefix):])
        return int(token)

    @classmethod
    def build(
        cls,
        train_ordered_names: Sequence[Sequence[str]],
        train_ordered_values: Sequence[Sequence[int]],
        all_ordered_names: Sequence[Sequence[str]],
        domain_values_by_name: Mapping[str, Sequence[int]] | None = None,
        pad_to_vocab_size: int | None = None,
    ) -> "ParamTokenizer":
        tokens: List[str] = [cls.PAD, cls.BOS, cls.UNK]
        token_seen = set(tokens)

        for names, values in zip(train_ordered_names, train_ordered_values):
            for name, value in zip(names, values):
                token = cls.value_to_token(name, value)
                if token not in token_seen:
                    token_seen.add(token)
                    tokens.append(token)

        if domain_values_by_name:
            for name in sorted(domain_values_by_name.keys()):
                for value in domain_values_by_name[name]:
                    token = cls.value_to_token(name, int(value))
                    if token not in token_seen:
                        token_seen.add(token)
                        tokens.append(token)

        # Optionally pad vocab with unreachable dummy tokens so the output
        # softmax / embedding table can be kept at a target size independent
        # of how many legal tokens were discovered. Dummies never match
        # value_to_token output (neither int strings nor auto_unroll_max_step$*),
        # so they can never be produced or masked-in.
        if pad_to_vocab_size is not None and len(tokens) < int(pad_to_vocab_size):
            target = int(pad_to_vocab_size)
            idx = 0
            while len(tokens) < target:
                dummy = f"__DUMMY_{idx}"
                idx += 1
                if dummy in token_seen:
                    continue
                token_seen.add(dummy)
                tokens.append(dummy)

        vars_: List[str] = ["<VAR_PAD>"]
        var_seen = set(vars_)
        for names in all_ordered_names:
            for name in names:
                if name not in var_seen:
                    var_seen.add(name)
                    vars_.append(name)

        return cls(
            ParamTokenizerState(
                token_to_id={tok: idx for idx, tok in enumerate(tokens)},
                id_to_token=tokens,
                var_to_id={name: idx for idx, name in enumerate(vars_)},
                id_to_var=vars_,
            )
        )

    def to_state_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "var_to_id": self.var_to_id,
            "id_to_var": self.id_to_var,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "ParamTokenizer":
        return cls(
            ParamTokenizerState(
                token_to_id=payload["token_to_id"],
                id_to_token=payload["id_to_token"],
                var_to_id=payload["var_to_id"],
                id_to_var=payload["id_to_var"],
            )
        )

    def encode_values(self, ordered_names: Sequence[str], ordered_values: Sequence[int]) -> List[int]:
        encoded: List[int] = []
        for name, value in zip(ordered_names, ordered_values):
            token = self.value_to_token(name, value)
            encoded.append(self.token_to_id.get(token, self.unk_id))
        return encoded

    def encode_var_names(self, ordered_names: Sequence[str]) -> List[int]:
        return [self.var_to_id[name] for name in ordered_names]

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token[idx] for idx in ids]

    def candidate_mask_from_values(
        self,
        var_name: str,
        candidate_values: Iterable[int],
        *,
        device: torch.device | None = None,
        allow_unk: bool = True,
    ) -> torch.Tensor:
        mask = torch.zeros(len(self.id_to_token), dtype=torch.bool, device=device)
        unknown_present = False
        for value in candidate_values:
            token = self.value_to_token(var_name, int(value))
            idx = self.token_to_id.get(token)
            if idx is None:
                unknown_present = True
                continue
            mask[idx] = True
        if allow_unk and unknown_present:
            mask[self.unk_id] = True
        return mask

    def pad_only_mask(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(len(self.id_to_token), dtype=torch.bool, device=device)
        mask[self.pad_id] = True
        return mask
