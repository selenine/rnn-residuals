import html
import os
import urllib.request
from dataclasses import asdict

import torch
import wandb
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from rnn_residuals.config import TrainConfig, TransformerConfig
from rnn_residuals.nn.layers import Transformer
from rnn_residuals.sample import generate

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare(
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    with open(path) as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train, n_val = int(0.9 * len(data)), int(0.95 * len(data))

    return data[:n_train], data[n_train:n_val], data[n_val:], chars


class RandomWindows(IterableDataset):
    def __init__(self, data: torch.Tensor, n_ctx: int, seed: int = 0) -> None:
        super().__init__()

        self.data = data
        self.n_ctx = n_ctx
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        g = torch.Generator().manual_seed(self.seed + (worker.id if worker else 0))

        while True:
            i = torch.randint(len(self.data) - self.n_ctx - 1, (1,), generator=g).item()
            yield self.data[i : i + self.n_ctx + 1]


def lr_lambda(cfg: TrainConfig):
    def fn(step: int) -> float:
        return min(1.0, (step + 1) / cfg.n_warmup)

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

    train_data, val_data, test_data, chars = load_shakespeare(train_cfg.data_dir)
    assert len(chars) == model_cfg.n_vocab, f"n_vocab must be {len(chars)}"

    loader = DataLoader(
        RandomWindows(train_data, model_cfg.n_ctx), batch_size=train_cfg.batch_size
    )
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(
        len(val_data) - model_cfg.n_ctx - 1,
        (train_cfg.eval_batches, train_cfg.batch_size),
        generator=g,
    )
    val_batches = [
        torch.stack([val_data[i : i + model_cfg.n_ctx + 1] for i in row])
        for row in starts.tolist()
    ]
    n_ctx = model_cfg.n_ctx
    test_windows = torch.stack(
        [
            test_data[i * n_ctx : (i + 1) * n_ctx + 1]
            for i in range((len(test_data) - 1) // n_ctx)
        ]
    )

    model = Transformer(model_cfg)

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

        if step % train_cfg.eval_every == 0 or step == train_cfg.n_batches:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    F.cross_entropy(
                        model(b[:, :-1].to(accelerator.device)).flatten(0, 1).float(),
                        b[:, 1:].flatten().to(accelerator.device),
                    ).item()
                    for b in val_batches
                ) / len(val_batches)
                test_nats = sum(
                    F.cross_entropy(
                        model(b[:, :-1].to(accelerator.device)).flatten(0, 1).float(),
                        b[:, 1:].flatten().to(accelerator.device),
                        reduction="sum",
                    ).item()
                    for b in test_windows.split(train_cfg.batch_size)
                ) / (len(test_windows) * n_ctx)
            model.train()

            accelerator.log({"val_loss": val_loss, "test_nats": test_nats}, step=step)
            if accelerator.is_main_process:
                pbar.write(
                    f"step {step} | val_loss {val_loss:.4g} | test_nats {test_nats:.4g}"
                )

                if train_cfg.sample_tokens > 0:
                    sample = generate(
                        accelerator.unwrap_model(model),
                        chars,
                        n_tokens=train_cfg.sample_tokens,
                    )
                    pbar.write(f"--- sample @ step {step} ---\n{sample}\n---")
                    accelerator.get_tracker("wandb", unwrap=True).log(
                        {"sample": wandb.Html(f"<pre>{html.escape(sample)}</pre>")},
                        step=step,
                    )

        if step % train_cfg.save_every == 0 or step == train_cfg.n_batches:
            accelerator.wait_for_everyone()
            os.makedirs(train_cfg.save_path, exist_ok=True)
            accelerator.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "cfg": asdict(model_cfg),
                    "chars": chars,
                    "step": step,
                },
                os.path.join(train_cfg.save_path, f"step_{step}.pt"),
            )

        if step == train_cfg.n_batches:
            break

    pbar.close()
    accelerator.end_training()
