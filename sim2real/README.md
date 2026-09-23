# Phase 3: Sim-to-Real

**Goal**: take a policy trained in this repo (RL or BC) and run it on your physical SO-101.

## Where this code runs

Everything else in this repo (`scripts/train.py`, `imitation/*`) runs on a GPU box with Isaac
Sim installed -- this AWS instance. Your SO-101 is a physical arm on a USB serial connection,
which this cloud box almost certainly cannot see. **The deployment code in this directory is
meant to run on whatever machine is physically plugged into the arm** (your laptop, a local
desktop, a Raspberry Pi next to it -- whatever you're using), not here. It needs a lightweight
env with just `torch` + LeRobot's SO-101 driver, not a full Isaac Sim install.

Workflow: train/export a policy here (or on your local sim setup) -> copy the exported
policy file (`.onnx` or `.pt`, see below) to the machine with the robot -> run `deploy_policy.py`
there.

## Concepts / tips

- **The sim-real gap is the whole problem.** Things that differ between sim and reality and
  will bite you: actuator dynamics (backlash, static friction stiction, PWM/torque nonlinearity
  the `sts3215` actuator's `stiffness`/`damping`/`friction` params in `assets/so101.py` are
  *already* an attempt to system-identify this -- check whether whoever tuned those values
  measured them from the real hardware, and if not, that's a good place to improve fidelity),
  sensor noise/latency, control-loop timing jitter, and anything about the environment (table
  height, exact robot mounting position) that isn't reproduced in the USD/URDF scene.
- **Domain randomization** is the standard mitigation: randomize sim physics/observations
  during training (already partially present -- e.g. `EventCfg.reset_robot_joints` randomizes
  initial joint positions) so the policy doesn't overfit to exact simulated dynamics. If sim2real
  transfer fails, "randomize more of the things that differ between sim and real" is the first
  lever to pull, before assuming the policy itself is bad.
- **Policy export**: IsaacLab's `play.py` (see `scripts/play.py` in this repo, and how the
  G1 demo's `runner.export_policy_to_jit`/`export_policy_to_onnx` calls work) already gives you
  a path to export a trained checkpoint to a standalone ONNX or TorchScript file with no Isaac
  Sim dependency -- that's the artifact you copy to the robot machine. Confirm this repo's own
  `runner.export_...` calls (if present) work for a `reach-v0` checkpoint the same way.
- **Observation matching is the easiest thing to get subtly wrong.** Your policy's input needs
  to be assembled identically on the real robot as it was in sim: same units, same joint
  ordering, same normalization -- check `ObservationsCfg.PolicyCfg` in `tasks/reach_env_cfg.py`
  term-by-term (`joint_pos_rel`, `joint_vel_rel`, `pose_command`, `last_action`) and make sure
  `deploy_policy.py` builds the exact same vector, in the same order, from real sensor readings.
- **Safety, before you run anything on real hardware:**
  - Clamp commanded actions to safe joint limits regardless of what the policy outputs -- don't
    trust a freshly-deployed policy.
  - Rate-limit / smooth commands (the RL action space is `RelativeJointPositionActionCfg`, i.e.
    *deltas* -- a bad delta compounds every timestep if unclamped).
  - Have a physical or software e-stop path, and test the control loop's watchdog/timeout
    behavior (what happens if inference stalls for a frame?) before trusting it near the arm.
  - Start with the gripper/end-effector far from anything breakable, and be ready to cut power.

## Files

Run from this directory on the Mac, in the order you would use them on a new arm. All talk
to the servos through `scservo_sdk` (`pip install feetech-servo-sdk`); LeRobot is not needed.

- `scan_motors.py` -- scan every common baud rate and list the servo IDs/models that answer
- `read_joints.py` -- live raw tick readout per joint, for hand-sweep checks
- `diag_servo.py` -- read one servo's control table and check write results (why won't it move?)
- `real_robot_interface.py` -- `SO101Bus`: tick<->radian mapping (651.9 ticks/rad, 2048 = 0),
  URDF joint clamps, checked writes, thermal monitoring. Motion is off unless `allow_motion=True`
- `goto_zero.py` -- ramp joints one at a time to the sim zero pose at 0.2 rad/s; dry run by
  default, `--allow-motion` to move, `--hold` to keep torque on afterwards
- `deploy_policy.py` -- run an exported `policy.onnx` at 50 Hz; dry run by default

Exported policies (`policy.onnx` + its `policy.onnx.data` weights) and run logs are gitignored.

## Status (2026-09-23)

- All six joints' signs and the tick->radian scale are verified on hardware by ramping each
  joint to zero. They settle 10-16 ticks short under load (P-only loop + deadband).
- The first reach policy does **not** transfer: even started from the exact zero pose, it
  swings back and forth in a regular ~2.4 s cycle instead of settling. The policy learned
  on sim joints that respond almost instantly (up to 0.25 rad per 20 ms step), while the
  real servos top out near 0.92 rad/s, so every step pushes against the 0.12 rad cap on
  how far the command may lead the arm. The fix is a retrain with realistic joint speed
  and action scale plus start-pose randomization, not a deploy-side change.

## Retrain: match the sim actuator to the real servos (2026-09-23)

What changed, and why, after the first policy oscillated on hardware:

| Change | Before | After | Why |
|---|---|---|---|
| `assets/so101.py` `velocity_limit_sim` | 10.0 rad/s | 0.92 rad/s | Real servos run at `GOAL_SPEED` 600 ticks/s = 0.92 rad/s. A ~10x faster sim arm taught the policy aggressive commands that overshoot on the slow real arm. |
| `ActionsCfg` `clip` | ±1.0 rad | ±0.25 rad | `clip` bounds the *scaled* offset. ±1.0 never engaged, so sim allowed offsets deploy never sends. ±scale matches deploy's `clip(action, -1, 1)`. |
| `ActionsCfg` `scale` | 0.25 | 0.25 (unchanged) | Shrinking it would break gravity holding: relative targets give torque = stiffness × offset, and 17.8 × 0.02 ≈ 0.36 Nm can't hold up ~0.7 Nm of arm. Speed is limited by the velocity cap instead. |
| Reset event | `reset_joints_by_scale` (0.5, 1.5) | `reset_joints_by_offset` (−1, 1) rad | Scaling an all-zero default pose always gave q=0, so the policy had never started anywhere else. |
| New `randomize_actuator_gains` | — | stiffness × U(0.7, 1.3) per env | Real servos are stiffer and vary; don't overfit one gain. |
| `deploy_policy.py` target rule | integrated setpoint, 0.02 rad/step clamp | `measured + clip(0.25·a)`, servo speed cap does the limiting | Same rule as training's `RelativeJointPositionAction`. |

A policy trained *before* this change is out of spec for the new deploy rule, and a new
policy is out of spec for the old one: re-export and deploy them together.

Training command (AWS box, from `~/IsaacLab`). Videos record every 250 iterations for one
full 12 s episode (600 steps) and upload to wandb alongside the curves:

    ./isaaclab.sh -p ~/projects/so101_isaac/scripts/train.py --task reach-v0 --headless \
        --num_envs 8192 --max_iterations 3000 --run_name actuator_match \
        --video --video_length 600 --video_interval 6000

Before retraining, `scripts/sim_check_real_limits.py` can confirm the diagnosis on the *old*
policy: `--mode sim` should settle, `--mode real` should oscillate like the hardware did.
