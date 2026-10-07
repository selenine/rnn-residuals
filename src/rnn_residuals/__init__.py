import argparse
from dataclasses import fields

import yaml

from rnn_residuals.config import TrainConfig, TransformerConfig
from rnn_residuals.train import train


def build(cls, raw: dict):
    types = {f.name: f.type for f in fields(cls)}
    return cls(**{k: float(v) if types.get(k) is float else v for k, v in raw.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    model_cfg = build(TransformerConfig, raw["model"])
    train_cfg = build(TrainConfig, raw["train"])

    train(model_cfg, train_cfg)
