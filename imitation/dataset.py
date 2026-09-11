"""Demonstration storage for behavioral cloning.

TODO(you): decide a concrete on-disk format and implement it here. Two reasonable options:

  1. Simplest: one .pt file per episode, each a dict of stacked tensors
     {"obs": (T, obs_dim), "action": (T, action_dim)}. Easy to reason about, fine for a
     baby project's dataset size.
  2. LeRobotDataset (https://github.com/huggingface/lerobot) if you want compatibility with
     the wider LeRobot ecosystem (their training scripts, visualizers, hub uploads, etc).
     More setup, more payoff if you continue this project past the class.

Either way, keep the *interface* below stable so collect_demos.py / train_bc.py don't care
which format you picked.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    """A dataset of (obs, action) pairs collected from demonstrations.

    TODO(you): implement __init__ to load whatever on-disk format you chose, and populate
    self.observations / self.actions as stacked tensors of shape (N, obs_dim) / (N, action_dim).

    Tip: keep episode boundaries around (e.g. self.episode_ends) even if you don't need them
    yet -- you'll want them if you ever move past plain BC to something history-aware (RNN/
    Transformer policies, ACT-style action chunking, etc), since you can't sample across an
    episode boundary for those.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        # TODO(you): load episodes from self.data_dir into these
        self.observations: torch.Tensor  # (N, obs_dim)
        self.actions: torch.Tensor  # (N, action_dim)
        raise NotImplementedError("TODO: implement dataset loading")

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


def save_episode(data_dir: str | Path, episode_idx: int, obs: torch.Tensor, actions: torch.Tensor) -> None:
    """Append one collected episode to disk.

    TODO(you): implement to match whatever format DemoDataset expects.
    obs: (T, obs_dim) float tensor. actions: (T, action_dim) float tensor.
    """
    raise NotImplementedError("TODO: implement episode saving")
