"""Train a behavioral-cloning policy on collected demonstrations.

No sim needed to run this one -- it's plain supervised learning over a saved dataset, so no
AppLauncher/Isaac Sim boilerplate. Run with the same isaaclab.sh python (has torch + wandb):

    ./isaaclab.sh -p ../imitation/train_bc.py --data_dir ./demos --epochs 100
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DemoDataset

parser = argparse.ArgumentParser(description="Train a BC policy for SO-101.")
parser.add_argument("--data_dir", type=str, default="./demos")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wandb", action="store_true", default=False)
args_cli = parser.parse_args()


class BCPolicy(nn.Module):
    """Minimal MLP policy: obs -> action.

    TODO(you): size the input/output dims to match your actual observation/action space
    (check reach_env_cfg.py's ObservationsCfg/ActionsCfg -- the PPO actor network printed at
    the start of training also tells you the exact in/out feature counts). Tune hidden_dims,
    activation, etc -- same knobs as the RL actor network, different training signal.

    Tip: this is deliberately the *same shape of problem* as the PPO actor you already tuned
    in Phase 1 (an MLP mapping obs -> action) -- the interesting new part isn't the network,
    it's the loss (supervised regression vs. policy gradient) and the data (demos vs. reward-
    driven rollouts). Don't over-engineer the network before the training loop works.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] = [128, 128]):
        super().__init__()
        raise NotImplementedError("TODO: build the MLP (nn.Sequential of Linear + activation)")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")


def train(args_cli) -> None:
    """TODO(you): standard supervised training loop.

    Skeleton:
      dataset = DemoDataset(args_cli.data_dir)
      loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=True)
      policy = BCPolicy(obs_dim=..., action_dim=...).to(device)
      optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)
      for epoch in range(args_cli.epochs):
          for obs, action in loader:
              pred = policy(obs)
              loss = F.mse_loss(pred, action)   # simplest starting loss
              optimizer.zero_grad(); loss.backward(); optimizer.step()
          # TODO: log loss (wandb if args_cli.wandb, matching the pattern from RL training),
          # periodically save a checkpoint (torch.save(policy.state_dict(), ...))

    Tips:
      - MSE loss assumes a unimodal, deterministic action per observation. If your demos have
        multiple valid ways to solve the task (e.g. multiple human demonstrators), a plain MSE
        policy will "average" between them into an invalid action -- one of the core reasons
        more advanced IL methods (mixture density nets, diffusion policies, ACT) exist. Not a
        concern for solo reach-task demos, worth knowing for later.
      - Normalize observations (and maybe actions) the same way your RL policy did -- check
        whether reach_env_cfg's ObservationsCfg has enable_corruption/noise you should turn off
        for eval, and whether you should apply the same obs normalization stats.
      - Split off a held-out validation set from your demos and track val loss, not just train
        loss -- BC overfits fast on small demo counts.
    """
    raise NotImplementedError("TODO: implement the training loop")


if __name__ == "__main__":
    train(args_cli)
