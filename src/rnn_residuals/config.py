from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_ctx: int
    n_vocab: int
    n_layers: int
    n_heads: int
    n_loops: int
    d_model: int
    d_head: int
    d_mlp: int
    causal: bool = True
    n_puzzles: int = 0
    grad_ckpt: bool = False
    n_grad_loops: int | None = None


@dataclass
class TrainConfig:
    lr: float
    n_warmup: int
    n_batches: int
    batch_size: int
    wt_decay: float
    grad_norm: float
    mixed_precision: str
    save_every: int
    save_path: str
    wandb_name: str
    log_every: int


@dataclass
class ArcConfig:
    data_dir: str
    n_aug: int
    n_eval_aug: int
    puzzle_lr: float
    puzzle_wt_decay: float
    eval_every: int
    eval_batch_size: int
