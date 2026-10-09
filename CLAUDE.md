# so101_isaac — project context

SO-101 robot arm learning project: **RL reach task in Isaac Lab → sim2real on the real arm →
imitation learning (leader/follower demos + ACT)**. Started from a UT Austin robotics class repo.
Details live in each folder's README; this file is the map.

## Which machine does what

| Machine | Role | Notes |
|---|---|---|
| **Mac** | Home of the repo; all hardware work | Both arms + wrist camera are USB here. `sim2real/` and `real_imitation/` only run here. |
| **Lab tower** (Windows, RTX 4060) | Screen/keyboard only | Cursor Remote-SSH into the Mac. No separate clone, no training. |
| **AWS box** (L40S) | Isaac Sim training | Repo at `~/projects/so101_isaac`, symlinked into `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/so101_isaac`. No SSH from the Mac: the user runs training there by hand. |

Code moves between the Mac and AWS through GitHub (`jatinsikka/so101_isaac`) only.

## Layout

- `assets/`, `tasks/`, `mdp/`, `agents/`: the `reach-v0` Isaac Lab task (robot, env config,
  rewards, PPO config). PPO comes from the class's `robot_rl` fork, not stock rsl_rl.
- `scripts/train.py`, `scripts/play.py`: training/export (run on AWS).
- `sim2real/`: servo interface + deploy of the exported ONNX policy (Mac, **base** conda env).
- `real_imitation/`: LeRobot leader→follower teleop/record/ACT (Mac, **`lerobot`** conda env).
- `imitation/`: sim behavioural-cloning scaffold, TODOs intentionally unfilled.

## Environments

- Mac base conda: Python 3.14, `feetech-servo-sdk`, onnxruntime. Used by `sim2real/`.
- Mac `conda activate lerobot`: Python 3.12 (LeRobot's draccus crashes on 3.14). Created from
  conda-forge only (`--override-channels -c conda-forge`; Anaconda ToS not accepted).
  `~/lerobot` is a clean editable clone: don't put project code in it.
- AWS: Isaac Lab **3.0**, newer than the class targeted. Run the repo's scripts by explicit path
  (`./isaaclab.sh -p ~/projects/so101_isaac/scripts/train.py`): a bare `train.py` resolves to
  Isaac Lab's bundled script.

## Hardware facts (verified)

- Follower `/dev/tty.usbmodem5AAF2879831` (12 V, has wrist camera = OpenCV index 0).
  Leader `/dev/tty.usbmodem5A7C1192341` (~5 V). Baud 1,000,000. Tell them apart by voltage.
- 6 × Feetech STS3215, motor IDs 1–6 = sim joint order: shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll, gripper. No remapping needed.
- 651.9 ticks/rad, tick 2048 = sim zero; every joint's sign verified 2026-09-23. LeRobot
  calibration kept that zero (offsets moved ≤ 2.5°).
- Real speed cap: `GOAL_SPEED` 600 ticks/s ≈ **0.92 rad/s** (`sim2real/real_robot_interface.py`).
- STS3215 load register: bit 10 (1024) is a direction flag, not magnitude.

## Policy I/O contract (reach-v0)

obs 25 = joint_pos_rel(6) + joint_vel_rel(6) + pose_command(7: xyz + quat wxyz, base frame)
+ last_action(6). action 6. 50 Hz. `target = current_pos + clip(0.25 * action)`, radians.
Every episode starts at q=0 on `main` (reset by scale on an all-zero default).

## Isaac Lab 3.0 gotchas already fixed

`UniformNoiseCfg` (not `AdditiveUniformNoiseCfg`); import `configclass` from
`isaaclab.utils.configclass`; `UrdfFileCfg(collision_type="Convex Hull")`, no
`make_instanceable`; `root/body_pos_w`/`_quat_w` need `.torch`; the table needs
`pos=(0.5,0,0), rot=(0,0,0.707,0.707)`, otherwise the robot is buried in its collision box.
Don't judge reward balance from the first ~10 iterations or small `num_envs`: squared penalties
look huge while the random policy thrashes and collapse later.

## Status (2026-10-08)

- **sim2real**: first policy oscillated on the real arm (limit cycle, `sim2real/live_run1.log`).
  Leading hypothesis: the sim actuator is ~10x faster than the real servo. But that log shows
  deploy's *filtered* targets (0.02 rad/step clamp, integrated setpoint, 0.12 rad lead cap), not
  the raw policy actions, and its `lead` column is pinned by the stuck gripper (closes at
  ≈ −0.034, sim limit −0.175). So the cause isn't proven yet.
- **Active branch `sim2real-step-by-step`** (one change per commit, **not trained yet**):
  velocity_limit_sim 10 → 0.92 rad/s, effort_limit_sim 10 → 2.9 Nm (stall torque), clip ±0.25,
  reset by offset (−1, 1), stiffness ×U(0.7, 1.3), deploy `target = measured + clip(0.25·a)`,
  GoalSpeed/Acceleration verified before torque-on, `scripts/sim_check_real_limits.py`.
  It supersedes `sim2real-actuator-match` (PR #2) and `sim-check-real-limits`.
- Next, on AWS: `sim_check --mode sim` and `--mode real` on the OLD policy (settles the
  diagnosis), then retrain, then `--mode match` on the new run, export ONNX, deploy from q=0.
- **real_imitation** (branch `real-imitation-demo`): calibration + teleop done; next is a
  2-episode trial `./04_record.sh 2`, then 50 episodes, then ACT training (the user writes that).

## How to work with this user

- Explain every change: small steps, show the diff and the reason. They want to understand it.
- It's their learning project: leave the core learning pieces as TODOs with hints unless they
  ask for an implementation. Real plumbing/compat bugs are fine to fix outright.
- Commit and push after each major step; commented, well-organized code.
- Don't browse personal files outside this repo without a concrete reason.
- Size training runs to saturate the L40S (large `num_envs`, headless).
