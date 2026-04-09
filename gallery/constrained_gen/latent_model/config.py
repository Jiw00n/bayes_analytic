from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_JSON_PATH = [
    # "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/584_([cb7a0e9e733d26ffc00e7f6c9cc0f879,[1,128,128,32],[1,1,32,16],[1,1,1,16],[1,128,128,16]],cuda).json",
    "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json"
]
DEFAULT_NETWORK_INFO_FOLDER = "/root/work/tvm-ansor/gallery/dataset/network_info_all"
DEFAULT_CHECKPOINT_DIR = "/root/work/tvm-ansor/gallery/constrained_gen/checkpoints"


@dataclass
class DataConfig:
    json_paths: List[str] = field(default_factory=lambda: list(DEFAULT_JSON_PATH))
    network_info_folder: str = DEFAULT_NETWORK_INFO_FOLDER
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42
    filter_invalid_records: bool = False


@dataclass
class ModelConfig:
    d_model: int = 192
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    # num_cost_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    latent_dim: int = 64
    latent_token_count: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    beta_start: float = 1e-4
    beta_end: float = 0.015
    beta_warmup_epochs: int = 10
    lambda_cost: float = 0.01
    lambda_nce: float = 0.2
    tau_nce: float = 0.2
    cost_ridge_vec: bool = True
    ridge_alpha: float = 0.1
    use_amp: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    allow_tf32: bool = False
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    resume_from: Optional[str] = None
    save_every_epoch: bool = True
    device: str = "cuda"
    debug_invalid_step: bool = True
    precompute_candidate_masks: bool = True
    order_nce: bool = True
    early_param_weight_max: float = 4.0
    early_param_weight_power: float = 1.5
    lambda_latent_use: float = 0.0
    latent_use_margin: float = 0.0
    latent_wrong_top1_margin: float = 0.0


@dataclass
class EvalConfig:
    greedy_decode: bool = True
    beam_size: int = 1
    batch_size: int = 128


@dataclass
class WandbConfig:
    project: Optional[str] = "single_v1"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
            eval=EvalConfig(**payload.get("eval", {})),
            wandb=WandbConfig(**payload.get("wandb", {})),
        )


CONFIG = ExperimentConfig()


def build_config() -> ExperimentConfig:
    return copy.deepcopy(CONFIG)
