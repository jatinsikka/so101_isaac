"""Evaluate a trained BC policy in sim before ever touching the real robot.

Same isaaclab.sh pattern as play.py -- reuse that file's AppLauncher/env-loading structure,
just swap in your BC policy instead of the rsl_rl runner.

    ./isaaclab.sh -p ../imitation/eval_bc.py --task reach-v0 --checkpoint ./bc_policy.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a BC policy for SO-101 reach.")
parser.add_argument("--task", type=str, default="reach-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401


def main() -> None:
    """TODO(you):
      1. Build the env the same way play.py does (gym.make(args_cli.task, cfg=...)).
      2. Load your BC policy's state dict (match the architecture from train_bc.py).
      3. Reset, then loop: action = policy(obs); obs, ... = env.step(action).
      4. Track something quantitative, not just "it looks okay" -- e.g. final position error
         to the commanded ee_pose per episode, success rate under some threshold. Compare it
         against your RL policy's numbers (same metric, same task) so "did IL work" has an
         actual answer instead of a vibe.

    Tip: if the BC policy flails where the RL policy didn't, suspect distribution shift first
    (see imitation/README.md) before assuming the network or training loop is wrong -- it's the
    single most common reason a low-training-loss BC policy fails at rollout time.
    """
    raise NotImplementedError("TODO: implement BC policy evaluation")


if __name__ == "__main__":
    main()
    simulation_app.close()
