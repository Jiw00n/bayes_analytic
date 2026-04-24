from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from glob import glob


# DEFAULT_JSON_PATHS = glob("/root/work/tvm-ansor/gallery/constrained_gen/data/measured_*/*.json")
DEFAULT_JSON_PATHS = glob("/root/work/tvm-ansor/gallery/dataset/measure_tenset_filtered_family/nn_contrib_conv2d_winograd_without_weight_transform/t4/*.json")
DEFAULT_NETWORK_INFO_FOLDER = "/root/work/tvm-ansor/gallery/dataset/network_info_all"
DEFAULT_CHECKPOINT_DIR = "/root/work/tvm-ansor/gallery/constrained_gen_v1.5_ori_shapes/checkpoints_all"


@dataclass
class DataConfig:
    json_paths: List[str] = field(default_factory=lambda: list(DEFAULT_JSON_PATHS))
    network_info_folder: str = DEFAULT_NETWORK_INFO_FOLDER
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42
    filter_invalid_records: bool = False
    budget: bool = False
    pad_vocab_to: Optional[int] = None


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
    seed: Optional[int] = 42
    vocab_align_to: Optional[int] = None


@dataclass
class TrainConfig:
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    precompute_candidate_masks: bool = True

    learning_rate: float = 5e-4
    beta_start: float = 1e-4
    beta_end: float = 0.003
    beta_warmup_epochs: int = 10
    lambda_recon: float = 1.0
    lambda_cost: float = 0.01
    lambda_nce: float = 0.2
    tau_nce: float = 0.2
    cost_ridge_vec: bool = True
    cost_ridge_include_val: bool = True
    ridge_alpha: float | List[float] = 0.1
    order_nce: bool = True
    order_nce_pos_weight_by_percentile: bool = False
    order_nce_pos_weight_sigma: float = 0.2
    nce_mu: bool = False
    
    latent_walk_top_k: int = 1
    latent_walk_num_steps: int = 30
    latent_walk_step_size: float = 0.25
    latent_walk_every_n_epochs: int = 10
    latent_walk_predict_every_n_epochs: int = 10

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

    cobo_sample_weighting: bool = False
    weight_quantile: float = 0.85
    weight_sigma: float = 0.25
    cobo_apply_to: List[str] = field(default_factory=lambda: ["kld", "cost", "nce"])
    cost_ridge_weighted: bool = False
    latent_walk_use_cost_head: bool = False

    # "cost_head" | "cost_vec" | "cost_vec_weighted" | "gp" | "lightgbm_ranker"
    re_encode_predictor: str = "cost_vec"
    latent_walk_predict_gp_top_k: int = 100
    latent_walk_predict_gp_random_n: int = 0
    latent_walk_predict_use_gp: bool = False


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
class SamplingConfig:
    """Decoding strategy used by `greedy_decode_*` and the latent walk.

    ``strategy`` selects the decoding mode. Any other value is treated as
    sampling with the listed truncation parameters.

    - ``strategy="greedy"``: argmax (ignores every other field).
    - ``strategy="sampling"``: multinomial over masked candidates after
      applying temperature, then optional top-k, then optional top-p.
    - ``top_k=0`` disables top-k; ``top_p=1.0`` disables top-p; the two can
      be combined (top-k first, then top-p).
    """

    strategy: str = "greedy"  # "greedy" | "sampling"
    temperature: float = 0.8
    top_k: int = 2
    top_p: float = 1.0
    seed: Optional[int] = 42


@dataclass
class WandbConfig:
    project: Optional[str] = "shapes"


@dataclass
class GeneratorConfig:
    """ScheduleGenerator overrides.

    ``hw_param`` patches entries in ``ScheduleGenerator.DEFAULT_HW_PARAM`` and
    ``disable_constraint`` removes kinds from
    ``ScheduleGenerator.DEFAULT_ENABLED_CONSTRAINT_KINDS``. Only values that
    differ from the defaults affect the precompute mask cache name.
    """

    hw_param: Dict[str, Any] = field(default_factory=dict)
    # hw_param: Dict[str, Any] = field(default_factory=lambda: {"max_vthread_extent": 15})


    disable_constraint: List[str] = field(default_factory=list)
    # disable_constraint: List[str] = field(default_factory=lambda: ["vectorize"])


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

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
            sampling=SamplingConfig(**payload.get("sampling", {})),
            wandb=WandbConfig(**payload.get("wandb", {})),
            generator=GeneratorConfig(**payload.get("generator", {})),
        )


CONFIG = ExperimentConfig()


def build_config() -> ExperimentConfig:
    return copy.deepcopy(CONFIG)
