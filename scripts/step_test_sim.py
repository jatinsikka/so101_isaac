"""step_test_sim.py -- the sim half of sim2real/step_test.py.   [C3, 2026-10-10]

Same test, same table: hold the arm at zero, move ONE joint's position target by a fixed
step, record 1 s, put it back, record 1 s. No policy involved; this drives the sim
joint's target directly, exactly like the real test drives the servo's goal position.
Uses the reach-v0 robot as configured (assets/so101.py actuator), so it shows how the
actuator the policy was trained on responds.

Run on the AWS box, from ~/IsaacLab:
    ./isaaclab.sh -p ~/projects/so101_isaac/scripts/step_test_sim.py --headless --joint elbow_flex
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--joint", default="elbow_flex")
parser.add_argument("--out", default="step_sim.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

STEPS = [0.01, 0.02, 0.05, 0.1, 0.2]  # rad, same as the real test
RECORD_S = 1.0

cfg = parse_env_cfg("reach-v0", num_envs=1)
cfg.events.reset_robot_joints = None  # start exactly at q = 0
env = gym.make("reach-v0", cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
j = robot.joint_names.index(args.joint)
physics_dt = env.sim.get_physics_dt()
steps_per_record = int(RECORD_S / env.step_dt)  # env.step_dt = 0.02 s, the 50 Hz policy rate


def record(target):
    """Hold `target` for RECORD_S; return (t, pos, torque) of joint j every 0.02 s."""
    rows = []
    for k in range(steps_per_record):
        for _ in range(env.cfg.decimation):  # 2 physics steps per 0.02 s, like training
            robot.set_joint_position_target(target)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(physics_dt)
        rows.append(((k + 1) * env.step_dt, robot.data.joint_pos[0, j].item(),
                     robot.data.applied_torque[0, j].item()))
    return rows


with torch.inference_mode():
    hold = torch.zeros(1, robot.num_joints, device=env.device)
    settle = record(hold)  # the sim joint holding "0" under gravity
    print(f"\n{args.joint} told to hold 0.000 rad, actually sits at {settle[-1][1]:+.3f} rad "
          f"(torque {settle[-1][2]:+.2f} Nm)")
    base = settle[-1][1]
    with open(args.out, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step_rad", "phase", "t", "target", "pos", "torque_nm"])
        print(f"\n{'step':>6} {'reached':>8} {'time to 90%':>12} {'back to':>8}")
        for step in STEPS:
            target = hold.clone()
            target[0, j] += step
            up, back = record(target), record(hold)
            for phase, rows, tgt in (("up", up, step), ("back", back, 0.0)):
                for t, q, tau in rows:
                    log.writerow([step, phase, f"{t:.3f}", f"{tgt:.4f}", f"{q:.4f}", f"{tau:.3f}"])
            moved = [q - base for _, q, _ in up]
            hit = [t for (t, _, _), m in zip(up, moved) if m >= 0.9 * step]
            t90 = f"{hit[0]:.2f} s" if hit else "never"
            print(f"{step:>6.2f} {moved[-1] / step:>7.0%} {t90:>12} {back[-1][1] - base:>+8.3f}")
    print(f"\nsaved {args.out}")

env.close()
app.close()
