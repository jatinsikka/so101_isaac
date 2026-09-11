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

- `deploy_policy.py` -- loads an exported policy and runs the real-time control loop against
  the physical SO-101 via LeRobot's driver (TODO: almost everything, see file)
- `real_robot_interface.py` -- thin wrapper around LeRobot's SO-101 driver exposing exactly the
  read-obs / send-action interface `deploy_policy.py` needs (TODO)
