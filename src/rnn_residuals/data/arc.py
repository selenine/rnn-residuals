import io
import json
import os
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

ARC_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
GRID = 30
SEQ_LEN = GRID * GRID
N_VOCAB = 12
PAD, EOS = 0, 1
IGNORE = -100


@dataclass
class Puzzle:
    name: str
    train: list[tuple[np.ndarray, np.ndarray]]
    test: list[tuple[np.ndarray, np.ndarray]]


def load_arc(data_dir: str) -> tuple[list[Puzzle], list[Puzzle]]:
    path = os.path.join(data_dir, "arc-agi-1.zip")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(ARC_URL, path)

    splits = {"training": [], "evaluation": []}
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            parts = name.split("/")
            if len(parts) != 4 or parts[1] != "data" or not name.endswith(".json"):
                continue

            raw = json.load(io.TextIOWrapper(zf.open(name)))
            pairs = {
                k: [
                    (
                        np.array(p["input"], dtype=np.uint8),
                        np.array(p["output"], dtype=np.uint8),
                    )
                    for p in raw[k]
                ]
                for k in ("train", "test")
            }
            splits[parts[2]].append(
                Puzzle(parts[3][:-5], pairs["train"], pairs["test"])
            )

    return splits["training"], splits["evaluation"]


class Augmentations:
    def __init__(self, n_puzzles: int, n_aug: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        self.n_aug = n_aug
        self.dihedral = rng.integers(0, 8, size=(n_puzzles, n_aug))
        self.perm = np.zeros((n_puzzles, n_aug, 10), dtype=np.uint8)
        self.perm[..., 1:] = np.argsort(rng.random((n_puzzles, n_aug, 9)), axis=-1) + 1

        self.dihedral[:, 0] = 0
        self.perm[:, 0] = np.arange(10)

    def apply(self, grid: np.ndarray, puzzle: int, aug: int) -> np.ndarray:
        d = self.dihedral[puzzle, aug]
        grid = np.rot90(grid, d % 4)
        if d >= 4:
            grid = grid.T

        return self.perm[puzzle, aug][grid]

    def invert(self, grid: np.ndarray, puzzle: int, aug: int) -> np.ndarray:
        grid = np.argsort(self.perm[puzzle, aug]).astype(np.uint8)[grid]

        d = self.dihedral[puzzle, aug]
        if d >= 4:
            grid = grid.T

        return np.rot90(grid, -(d % 4))


def encode(grid: np.ndarray, r0: int = 0, c0: int = 0) -> np.ndarray:
    h, w = grid.shape
    seq = np.full((GRID, GRID), PAD, dtype=np.int64)

    seq[r0 : r0 + h, c0 : c0 + w] = grid.astype(np.int64) + 2
    if r0 + h < GRID:
        seq[r0 + h, c0 : c0 + w] = EOS
    if c0 + w < GRID:
        seq[r0 : r0 + h, c0 + w] = EOS

    return seq.reshape(-1)


def decode(seq: np.ndarray) -> np.ndarray | None:
    grid = seq.reshape(GRID, GRID)
    is_colour = grid >= 2

    w = GRID if is_colour[0].all() else int(np.argmin(is_colour[0]))
    h = GRID if is_colour[:, 0].all() else int(np.argmin(is_colour[:, 0]))
    if h == 0 or w == 0 or not is_colour[:h, :w].all():
        return None

    return (grid[:h, :w] - 2).astype(np.uint8)


class ArcTrain(Dataset):
    def __init__(
        self,
        train: list[Puzzle],
        evaluation: list[Puzzle],
        augs: Augmentations,
        translate: bool = True,
    ) -> None:
        self.augs = augs
        self.translate = translate
        self.pairs = [
            (i, x, y)
            for i, p in enumerate(train + evaluation)
            for x, y in p.train + (p.test if i < len(train) else [])
        ]

    def __len__(self) -> int:
        return len(self.pairs) * self.augs.n_aug

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        (puzzle, x, y), aug = self.pairs[idx // self.augs.n_aug], idx % self.augs.n_aug
        x, y = self.augs.apply(x, puzzle, aug), self.augs.apply(y, puzzle, aug)

        r0 = c0 = 0
        if self.translate:
            r0 = np.random.randint(0, GRID - max(x.shape[0], y.shape[0]) + 1)
            c0 = np.random.randint(0, GRID - max(x.shape[1], y.shape[1]) + 1)

        labels = encode(y, r0, c0)
        labels[labels == PAD] = IGNORE

        return {
            "tokens": torch.from_numpy(encode(x, r0, c0)),
            "labels": torch.from_numpy(labels),
            "puzzle_ids": torch.tensor(puzzle * self.augs.n_aug + aug),
        }


class ArcEval(Dataset):
    def __init__(
        self,
        n_train: int,
        evaluation: list[Puzzle],
        augs: Augmentations,
        n_eval_aug: int,
    ) -> None:
        self.augs = augs
        self.n_eval_aug = n_eval_aug
        self.tests = [
            (n_train + i, t, x, y)
            for i, p in enumerate(evaluation)
            for t, (x, y) in enumerate(p.test)
        ]

    def __len__(self) -> int:
        return len(self.tests) * self.n_eval_aug

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        test, aug = idx // self.n_eval_aug, idx % self.n_eval_aug
        puzzle, _, x, _ = self.tests[test]

        return {
            "tokens": torch.from_numpy(encode(self.augs.apply(x, puzzle, aug))),
            "puzzle_ids": torch.tensor(puzzle * self.augs.n_aug + aug),
            "index": torch.tensor(idx),
        }

    def score(self, preds: dict[int, np.ndarray]) -> dict[str, float]:
        per_puzzle: dict[int, list[tuple[bool, bool]]] = {}

        for test, (puzzle, _, _, y) in enumerate(self.tests):
            votes: Counter = Counter()
            grids = {}
            for aug in range(self.n_eval_aug):
                grid = decode(preds[test * self.n_eval_aug + aug])
                if grid is None:
                    continue

                grid = self.augs.invert(grid, puzzle, aug)
                key = (grid.shape, grid.tobytes())
                votes[key] += 1
                grids[key] = grid

            top = [grids[k] for k, _ in votes.most_common(2)]
            hits = [g.shape == y.shape and (g == y).all() for g in top]
            per_puzzle.setdefault(puzzle, []).append((any(hits[:1]), any(hits[:2])))

        return {
            f"pass@{k}": float(
                np.mean([np.mean([h[k - 1] for h in hs]) for hs in per_puzzle.values()])
            )
            for k in (1, 2)
        }
