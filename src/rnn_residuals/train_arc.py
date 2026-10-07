import os
from dataclasses import asdict

import torch
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn_residuals.config import ArcConfig, TrainConfig, TransformerConfig
from rnn_residuals.data import arc
from rnn_residuals.nn.layers import LoopedTransformer
from rnn_residuals.train import lr_lambda


@torch.no_grad()
def evaluate(
    model: LoopedTransformer, dataset: arc.ArcEval, batch_size: int, device
) -> dict[str, float]:
    model.eval()

    preds = {}
    for batch in DataLoader(dataset, batch_size=batch_size):
        logits = model(batch["tokens"].to(device), batch["puzzle_ids"].to(device))
        for i, seq in zip(batch["index"].tolist(), logits.argmax(-1).cpu().numpy()):
            preds[i] = seq

    model.train()
    return dataset.score(preds)


def train(
    model_cfg: TransformerConfig, train_cfg: TrainConfig, arc_cfg: ArcConfig
) -> None:
    accelerator = Accelerator(
        mixed_precision=train_cfg.mixed_precision, log_with="wandb"
    )
    accelerator.init_trackers(
        "rnn-residuals",
        config={
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
            "arc": asdict(arc_cfg),
        },
        init_kwargs={"wandb": {"name": train_cfg.wandb_name}},
    )

    train_puzzles, eval_puzzles = arc.load_arc(arc_cfg.data_dir)
    augs = arc.Augmentations(len(train_puzzles) + len(eval_puzzles), arc_cfg.n_aug)
    train_set = arc.ArcTrain(train_puzzles, eval_puzzles, augs)
    eval_set = arc.ArcEval(len(train_puzzles), eval_puzzles, augs, arc_cfg.n_eval_aug)

    model_cfg.n_puzzles = len(augs.dihedral) * arc_cfg.n_aug
    model = LoopedTransformer(model_cfg)

    puzzle = [model.puzzle_embed.weight]
    decay = [
        p for n, p in model.named_parameters() if p.dim() >= 2 and "puzzle" not in n
    ]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.wt_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {
                "params": puzzle,
                "lr": arc_cfg.puzzle_lr,
                "weight_decay": arc_cfg.puzzle_wt_decay,
            },
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(train_cfg))
    loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )

    model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)
    model.train()

    pbar = tqdm(
        total=train_cfg.n_batches, disable=not accelerator.is_local_main_process
    )
    step = 0
    while step < train_cfg.n_batches:
        for batch in loader:
            step += 1
            labels = batch["labels"]

            logits = model(batch["tokens"], batch["puzzle_ids"])
            loss = F.cross_entropy(
                logits.flatten(0, 1).float(), labels.flatten(), ignore_index=arc.IGNORE
            )

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), train_cfg.grad_norm
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.update()
            if step % train_cfg.log_every == 0:
                mask = labels != arc.IGNORE
                correct = (logits.argmax(-1) == labels) | ~mask
                stats = {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": sched.get_last_lr()[0],
                    "token_acc": (correct & mask).sum().item() / mask.sum().item(),
                    "exact_acc": correct.all(-1).float().mean().item(),
                }
                accelerator.log(stats, step=step)
                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}", exact=f"{stats['exact_acc']:.3f}"
                )

            if step % arc_cfg.eval_every == 0 or step == train_cfg.n_batches:
                if accelerator.is_main_process:
                    scores = evaluate(
                        accelerator.unwrap_model(model),
                        eval_set,
                        arc_cfg.eval_batch_size,
                        accelerator.device,
                    )
                    accelerator.log(
                        {f"eval/{k}": v for k, v in scores.items()}, step=step
                    )
                    pbar.write(f"step {step}: {scores}")
                accelerator.wait_for_everyone()

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


def main() -> None:
    model_cfg = TransformerConfig(
        n_ctx=arc.SEQ_LEN + 1,
        n_vocab=arc.N_VOCAB,
        n_layers=4,
        n_heads=16,
        n_loops=1024,
        d_model=512,
        d_head=32,
        d_mlp=2048,
        causal=False,
        grad_ckpt=True,
    )
    train_cfg = TrainConfig(
        lr=1e-4,
        n_warmup=2000,
        n_batches=100_000,
        batch_size=32,
        wt_decay=0.1,
        grad_norm=1.0,
        mixed_precision="bf16",
        save_every=10_000,
        save_path="checkpoints/arc",
        wandb_name="gdn2-arc-l2-x8",
        log_every=10,
    )
    arc_cfg = ArcConfig(
        data_dir="data",
        n_aug=1000,
        n_eval_aug=64,
        puzzle_lr=1e-2,
        puzzle_wt_decay=0.1,
        eval_every=10_000,
        eval_batch_size=64,
    )

    train(model_cfg, train_cfg, arc_cfg)
