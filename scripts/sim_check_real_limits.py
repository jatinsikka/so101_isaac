"""Does the trained policy still converge in sim once the arm is as slow as the real one?

On hardware the reach policy oscillated (~2.4 s period) instead of settling, even from the
zero pose. The hypothesis is an actuator speed mismatch: in sim a joint can move 0.25 rad per
20 ms step, the real STS3215s are capped near 0.92 rad/s. This reproduces the real test in
sim, so the hypothesis can be confirmed before spending a retrain on it.

    --mode sim    the ORIGINAL training physics: 10 rad/s joints, +-1 rad offset clip
                  (baseline for the first policy -- should converge)
    --mode real   the ORIGINAL deploy: 0.92 rad/s joints, 0.02 rad/step clamp
                  (what the first hardware run did -- expected to oscillate)
    --mode match  the actuator-matched setup: 0.92 rad/s joints, +-0.25 rad clip, which is
                  both the new training config and the new deploy rule (for the retrained policy)

Every mode sets these explicitly, so the result does not depend on which branch's defaults
are checked out.

Every mode starts every env at the zero pose with the same fixed target the real run used,
(0.27, 0, 0.15) in the base frame, and report end-effector error and joint oscillation.
If `real` oscillates like the hardware did and `sim` does not, the diagnosis holds.

Run from the IsaacLab directory (a bare `play.py`-style filename would be aliased):
    ./isaaclab.sh -p ~/projects/so101_isaac/scripts/sim_check_real_limits.py \
        --task reach-v0 --headless --mode sim  --load_run <run_folder>
    ./isaaclab.sh -p ~/projects/so101_isaac/scripts/sim_check_real_limits.py \
        --task reach-v0 --headless --mode real --load_run <run_folder>
    # after retraining, on the new run:
    ./isaaclab.sh -p ~/projects/so101_isaac/scripts/sim_check_real_limits.py \
        --task reach-v0 --headless --mode match --load_run <new_run_folder>

Approximations vs the real deploy loop: sim bases each target on the measured position
(RelativeJointPositionAction), while deploy_policy integrates a setpoint with a 0.12 rad
anti-windup lead. Servo deadband and serial latency are not modelled.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Check the reach policy under real-arm actuator limits.")
parser.add_argument("--task", type=str, default="reach-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--mode", choices=["sim", "real", "match"], required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=10.0, help="matches the real 10 s run")
parser.add_argument("--target", type=float, nargs=3, default=(0.27, 0.0, 0.15), metavar=("X", "Y", "Z"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from robot_rl.runners import OnPolicyRunner

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    # Same situation as the hardware run: one fixed target, held for the whole run.
    x, y, z = args_cli.target
    r = env_cfg.commands.ee_pose.ranges
    r.pos_x, r.pos_y, r.pos_z = (x, x), (y, y), (z, z)
    env_cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.episode_length_s = args_cli.seconds + 5.0  # no timeout reset mid-measurement
    # Start every env at q=0 like the hardware test, with nominal actuator gains.
    env_cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    if hasattr(env_cfg.events, "randomize_actuator_gains"):
        env_cfg.events.randomize_actuator_gains = None

    # (joint velocity limit rad/s, per-step offset clip rad). JointAction clips the *processed*
    # (scaled) action and last_action observes the raw one -- the same split deploy_policy.py has.
    max_joint_vel, max_delta = {"sim": (10.0, 1.0), "real": (0.92, 0.02), "match": (0.92, 0.25)}[args_cli.mode]
    for act in env_cfg.scene.robot.actuators.values():
        act.velocity_limit_sim = max_joint_vel
    env_cfg.actions.arm_action.clip = {".*": (-max_delta, max_delta)}
    print(f"[INFO] mode {args_cli.mode}: velocity_limit_sim={max_joint_vel} rad/s, offset clip=+-{max_delta} rad")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                 "logs", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    ee_id = robot.find_bodies("gripper_link")[0][0]
    steps = int(args_cli.seconds / uenv.step_dt)

    def ee_error() -> torch.Tensor:
        ee_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                                            robot.data.body_pos_w.torch[:, ee_id])
        return (ee_b - uenv.command_manager.get_command("ee_pose")[:, :3]).norm(dim=1)

    obs = env.get_observations()
    joint_hist, err_hist = [], []
    print(f"\n=== mode {args_cli.mode}: {args_cli.num_envs} envs, target {tuple(args_cli.target)} ===")
    print(f"{'t':>6} {'ee err mean':>12} {'ee err max':>11}   joint pos env0 (rad)")
    with torch.inference_mode():
        for step in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            err = ee_error()
            joint_hist.append(robot.data.joint_pos.clone())
            err_hist.append(err.clone())
            if step % 25 == 0:
                q = " ".join(f"{v:+.2f}" for v in joint_hist[-1][0].tolist())
                print(f"{step * uenv.step_dt:>6.2f} {err.mean():>12.4f} {err.max():>11.4f}   {q}")

    # Settled vs oscillating: look at the second half, after any initial transient.
    half = steps // 2
    q_tail = torch.stack(joint_hist[half:])        # (T, envs, joints)
    e_tail = torch.stack(err_hist[half:])          # (T, envs)
    swing = (q_tail.max(0).values - q_tail.min(0).values).mean(0)  # per-joint peak-to-peak, env-avg
    print(f"\n--- last {args_cli.seconds / 2:.0f} s ---")
    print(f"ee error: mean {e_tail.mean():.4f} m, final {e_tail[-1].mean():.4f} m")
    print("joint peak-to-peak (rad): " + "  ".join(f"{v:.3f}" for v in swing.tolist()))
    settled = e_tail[-1].mean() < 0.02 and swing.max() < 0.05
    print("verdict: " + ("SETTLED" if settled else "NOT SETTLED (still moving / oscillating)"))
    # For reference, the hardware run swung shoulder_pan ~0.6 rad and wrist_roll ~1.8 rad p-p.

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
