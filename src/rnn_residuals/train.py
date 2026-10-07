import math
import os
from dataclasses import asdict

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from rnn_residuals.config import TrainConfig, TransformerConfig
from rnn_residuals.nn.layers import LoopedTransformer


class PackedTokens(IterableDataset):
    def __init__(
        self,
        tokenizer,
        n_ctx: int,
        dataset: str = "roneneldan/TinyStories",
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.dataset = dataset

    def __iter__(self):
        ds = load_dataset(self.dataset, split="train", streaming=True)
        ds = ds.shuffle(seed=0, buffer_size=10_000)

        buf = []
        for ex in ds:
            buf.extend(self.tokenizer(ex["text"])["input_ids"])
            buf.append(self.tokenizer.eos_token_id)

            while len(buf) >= self.n_ctx + 1:
                yield torch.tensor(buf[: self.n_ctx + 1])
                buf = buf[self.n_ctx + 1 :]


def lr_lambda(cfg: TrainConfig):
    def fn(step: int) -> float:
        if step < cfg.n_warmup:
            return (step + 1) / cfg.n_warmup

        t = (step - cfg.n_warmup) / max(1, cfg.n_batches - cfg.n_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return fn


def train(model_cfg: TransformerConfig, train_cfg: TrainConfig) -> None:
    accelerator = Accelerator(
        mixed_precision=train_cfg.mixed_precision, log_with="wandb"
    )
    accelerator.init_trackers(
        "rnn-residuals",
        config={"model": asdict(model_cfg), "train": asdict(train_cfg)},
        init_kwargs={"wandb": {"name": train_cfg.wandb_name}},
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    loader = DataLoader(
        PackedTokens(tokenizer, model_cfg.n_ctx), batch_size=train_cfg.batch_size
    )

    model = LoopedTransformer(model_cfg)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.wt_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(train_cfg))

    model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)
    model.train()

    pbar = tqdm(
        total=train_cfg.n_batches, disable=not accelerator.is_local_main_process
    )
    for step, batch in enumerate(loader, start=1):
        x, y = batch[:, :-1], batch[:, 1:]

        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), train_cfg.grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        pbar.update()
        if step % train_cfg.log_every == 0:
            stats = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": sched.get_last_lr()[0],
            }
            accelerator.log(stats, step=step)
            if accelerator.is_main_process:
                pbar.write(
                    f"step {step} | "
                    + " | ".join(f"{k} {v:.4g}" for k, v in stats.items())
                )
            pbar.set_postfix(loss=f"{stats['loss']:.4f}")

        if step % train_cfg.save_every == 0 or step == train_cfg.n_batches:
            accelerator.wait_for_everyone()
            os.makedirs(train_cfg.save_path, exist_ok=True)
            accelerator.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "cfg": asdict(model_cfg),
                    "step": step,
                },
                os.path.join(train_cfg.save_path, f"step_{step}.pt"),
            )

        if step == train_cfg.n_batches:
            break

    pbar.close()
    accelerator.end_training()
