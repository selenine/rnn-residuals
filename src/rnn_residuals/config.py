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
    residual: str = "gdn2"
    causal: bool = True
    grad_ckpt: bool = False
    n_grad_loops: int | None = None
    use_alpha: bool = True
    use_beta: bool = True
    use_gamma: bool = True


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
    data_dir: str
    eval_every: int
    eval_batches: int
