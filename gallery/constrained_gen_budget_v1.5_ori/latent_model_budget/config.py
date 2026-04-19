from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_JSON_PATH = [
    # "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/584_([cb7a0e9e733d26ffc00e7f6c9cc0f879,[1,128,128,32],[1,1,32,16],[1,1,1,16],[1,128,128,16]],cuda).json",
    # "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json",
    "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json"
]
DEFAULT_NETWORK_INFO_FOLDER = "/root/work/tvm-ansor/gallery/dataset/network_info_all"
DEFAULT_CHECKPOINT_DIR = "/root/work/tvm-ansor/gallery/constrained_gen_budget_v1.5_ori/checkpoints_all/415"


@dataclass
class DataConfig:
    json_paths: List[str] = field(default_factory=lambda: list(DEFAULT_JSON_PATH))
    network_info_folder: str = DEFAULT_NETWORK_INFO_FOLDER
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42
    filter_invalid_records: bool = False
    budget: bool = False


@dataclass
class ModelConfig:
    d_model: int = 192
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    # num_cost_layers: int = 3
    cost_hidden_dim: int = 128
    dim_feedforward: int = 384
    dropout: float = 0.1
    latent_dim: int = 64
    latent_token_count: int = 4
    adaln: bool = True


@dataclass
class TrainConfig:
    learning_rate: float = 7e-4
    beta_start: float = 1e-4
    beta_end: float = 0.003
    beta_warmup_epochs: int = 20
    lambda_recon: float = 1.0
    lambda_cost: float = 0.01
    lambda_nce: float = 0.2
    tau_nce: float = 0.3
    cost_ridge_vec: bool = True
    ridge_alpha: float | List[float] = 0.1
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    precompute_candidate_masks: bool = True
    order_nce: bool = True
    nce_mu: bool = False
    latent_walk_top_k: int = 1
    latent_walk_num_steps: int = 30
    latent_walk_step_size: float = 0.25
    latent_walk_every_n_epochs: int = 10
    latent_walk_predict_every_n_epochs: int = 10
    latent_walk_predict_use_gp: bool = True
    latent_walk_predict_gp_top_k: int = 800
    latent_walk_predict_gp_random_n: int = 200
    label_smoothing: float = 0.0


    batch_size: int = 128
    num_epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 1000
    early_stop_min_delta: float = 1e-4

    best_metric_name: str = "walk/best_measured_mean_cost"
    best_metric_mode: str = "max"

    scheduler_name: str = "cosine"   # "none" | "multistep" | "plateau" | "cosine"
    cosine_t_max: int = 0
    cosine_eta_min: float = 1e-5
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1



    # scheduler_milestones: List[int] = field(default_factory=lambda: [30, 50])
    # scheduler_gamma: float = 0.5
    # plateau_factor: float = 0.5
    # plateau_patience: int = 5
    # plateau_threshold: float = 1e-4
    # plateau_min_lr: float = 1e-5


    latent_walk_on_final: bool = False
    latent_walk_record_json: Optional[str] = None
    latent_walk_output_dir: Optional[str] = None
    lambda_latent_use: float = 0.0
    latent_use_margin: float = 0.0
    latent_wrong_top1_margin: float = 0.0
    use_amp: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    allow_tf32: bool = False
    resume_from: Optional[str] = None
    device: str = "cuda"
    debug_invalid_step: bool = False
    use_compressed_teacher_forcing: bool = False

@dataclass
class EvalConfig:
    greedy_decode: bool = True
    beam_size: int = 1
    batch_size: int = 128
    final_full_autoregressive: bool = False


@dataclass
class WandbConfig:
    project: Optional[str] = "v1.5_ori"


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
