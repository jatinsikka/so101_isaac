"""Collect demonstrations for imitation learning.

Two collection modes -- implement whichever you want first (rollout is easier to start with
since it needs no hardware; teleop is the one that actually teaches you something new and
gives you a dataset an RL reward function couldn't).

Run from so101_isaac/scripts, same pattern as train.py:
    ./isaaclab.sh -p ../imitation/collect_demos.py --task reach-v0 --mode rollout --checkpoint <path>
    ./isaaclab.sh -p ../imitation/collect_demos.py --task reach-v0 --mode teleop

Tip: use an explicit relative/absolute path (e.g. `./collect_demos.py` or the full path) when
invoking scripts through `isaaclab.sh -p`, not a bare filename -- IsaacLab's CLI aliases bare
`train.py`/`play.py` to its own bundled scripts regardless of your cwd, which will silently run
the wrong file. Not an issue for `collect_demos.py` itself (no name collision), but worth knowing.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect demonstrations for SO-101 imitation learning.")
parser.add_argument("--task", type=str, default="reach-v0")
parser.add_argument("--mode", type=str, choices=["rollout", "teleop"], required=True)
parser.add_argument("--checkpoint", type=str, default=None, help="RL checkpoint, required for --mode rollout")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--out_dir", type=str, default="./demos")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401  (registers reach-v0 etc via import side-effect)

from dataset import save_episode  # noqa: E402


def collect_rollout(args_cli) -> None:
    """Roll out a trained RL policy and record (obs, action) pairs.

    TODO(you):
      1. gym.make(args_cli.task, ...) and load the policy from args_cli.checkpoint
         (see scripts/play.py in this repo for the loading pattern -- reuse it).
      2. For each episode: reset env, step with policy(obs) until done, stack obs/actions,
         call save_episode(...).
      3. Consider: do you want the *exact* RL rollout, or do you want some exploration
         noise added? Pure exploitation gives a narrower, more "expert" but less robust
         dataset -- worth experimenting with once the pipeline works end to end.
    """
    raise NotImplementedError("TODO: implement RL-policy rollout collection")


def collect_teleop(args_cli) -> None:
    """Collect demonstrations by teleoperating the arm.

    TODO(you): decide sim-teleop vs real-teleop:
      - Sim teleop: drive the simulated SO-101's `ee_pose` command (or joint targets directly)
        with e.g. a keyboard/gamepad/spacemouse device via IsaacLab's device interfaces
        (see isaaclab.devices -- Se3Keyboard etc are a reasonable starting point), record
        (obs, action) each env.step().
      - Real teleop: since you have the hardware, LeRobot's leader-follower teleop pattern
        (a second, ungeared "leader" SO-101 arm you move by hand, whose joint positions drive
        the "follower" arm) is the standard approach for this exact robot. See LeRobot's docs/
        examples for SO-101 teleop. You'd record real robot state/actions directly, no sim
        involved -- this produces demos with real-world dynamics/friction "for free", which
        sim rollouts can't give you, at the cost of needing the physical setup.
      - Either way: log at a fixed control rate consistent with what your policy will run at.
    """
    raise NotImplementedError("TODO: implement teleoperated demonstration collection")


def main():
    if args_cli.mode == "rollout":
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required for --mode rollout")
        collect_rollout(args_cli)
    else:
        collect_teleop(args_cli)


if __name__ == "__main__":
    main()
    simulation_app.close()
