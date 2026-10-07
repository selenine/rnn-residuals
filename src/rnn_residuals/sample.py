import argparse

import torch

from rnn_residuals.config import TransformerConfig
from rnn_residuals.nn.layers import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    chars: list[str],
    prompt: str = "\n",
    n_tokens: int = 500,
    temperature: float = 1.0,
) -> str:
    stoi = {c: i for i, c in enumerate(chars)}
    device = next(model.parameters()).device
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)

    was_training = model.training
    model.eval()
    for _ in range(n_tokens):
        logits = model(idx[:, -model.cfg.n_ctx :])[:, -1].float()
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial((logits / temperature).softmax(-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train(was_training)

    return "".join(chars[i] for i in idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--n-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = TransformerConfig(**ckpt["cfg"])

    model = Transformer(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    print(generate(model, ckpt["chars"], args.prompt, args.n_tokens, args.temperature))
