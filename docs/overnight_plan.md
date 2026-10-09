# Overnight plan (2026-10-08 → back ~20 h later)

Written on the Mac for the Claude Code session that runs **on the AWS box**. Read `CLAUDE.md`
first. The user is away the whole time: nobody will answer questions, so make the call
yourself, write down why in `docs/overnight_status.md`, and keep going.

Everything runs from branch **`multi-arm-push`** (it contains all of `sim2real-step-by-step`).
Use one `tmux` session per long job, so the jobs survive if the SSH session drops.

## Ground rules

- **Part 1 has priority over Part 2.** The reach retrain is the real goal; the multi-arm cell is a bonus.
- Both jobs share the L40S. Isaac Sim can run two processes on one GPU; watch `nvidia-smi`
  for memory headroom (48 GB).
- Keep `docs/overnight_status.md` up to date: what ran, the exact commands, run folder
  names, wandb links, verdicts, problems hit and how you fixed them. Commit and push it
  after each milestone, so the user can read it from the Mac.
- Commit code in small, commented commits and push (the user reviews every change later).
  No force pushes. Don't merge anything into `main`.
- Never judge reward balance from the first ~10 iterations or a tiny `num_envs` (see
  `CLAUDE.md`).
- Don't shut down or stop the instance.

## Part 1: reach sim2real (do this first, about 30 min of your attention)

1. **Diagnosis check on the OLD policy.** Find the old run under `logs/so101_reach/` (the
   one with `model_999.pt`). Run `scripts/sim_check_real_limits.py` with `--mode sim`, then
   `--mode real` (exact commands are in that script's docstring). Record both verdict blocks
   in the status file.
   - `sim` SETTLED, `real` NOT SETTLED → the diagnosis holds, so go ahead with step 2.
   - Both SETTLED → speed alone doesn't explain the hardware oscillation. **Still run step
     2** (the matched actuator is more faithful either way), but flag this prominently in
     the status file.
   - `sim` NOT SETTLED → the script or the checkpoint is wrong. Debug it (time-box 1 h),
     record what you find, and run step 2 anyway.
2. **Retrain** in tmux session `reach`:
   ```
   ./isaaclab.sh -p ~/projects/so101_isaac/scripts/train.py --task reach-v0 --headless \
       --num_envs 8192 --max_iterations 3000 --run_name actuator_match \
       --video --video_length 600 --video_interval 6000
   ```
3. When it finishes: run `sim_check_real_limits.py --mode match` on the new run, then
   export with `scripts/play.py --convert` (gives `exported/policy.onnx` and
   `policy.onnx.data`). Record the paths. The user copies these to the Mac by hand.

## Part 2: three-arm cooperative push cell (`multi-arm-push-v0`)

**The task.** Three SO-101 arms stand around a table at 120° spacing, all facing the
centre. A cube lies near the centre, and a goal spot is sampled anywhere in a disk around
the centre. One centralized PPO policy drives all 18 joints and has to push the cube onto
the goal. Why it needs cooperation: an arm can push *away* from itself but can't pull
toward itself. So depending on where the goal is, a different arm (or a pair) has to do
the work, and the others have to stay out of the way.

The user asked for this to be **fully implemented** so it can train tonight. The usual
"leave learning TODOs" rule is suspended for this task, but comment the reasoning
thoroughly so they can follow it later.

### Files

- `tasks/multi_arm_push_env_cfg.py`: the env config (follow the structure of `reach_env_cfg.py`)
- `mdp/multi_arm.py`: any custom observation, reward, termination and command terms
- `tasks/__init__.py` and the root `__init__.py`: register `multi-arm-push-v0`
- `agents/rsl_rl_ppo_cfg.py`: `MultiArmPushPPORunnerCfg`, `experiment_name = "so101_multi_arm_push"`,
  `wandb_project = "SO101_MultiArm"`

### Scene (per env; env origin = cell centre)

- **Table**: a static `CuboidCfg` slab, 1.2 × 1.2 × 0.05 m, top at z = 0. Use this, not the
  SeattleLabTable USD: it has known geometry, and it avoids the table-rotation trap.
- **Arms** `robot_a/b/c` = `SO101_CFG.replace(prim_path=..., init_state=...)`. Bases at radius
  **0.26 m**, angles θ = 90°, 210°, 330°, base z = 0.01, yaw = θ + 180° (the base's +x is the
  reach direction, so this points each arm at the centre).
  **Verify the quaternion convention** (Isaac Lab 3.0 may differ from the old wxyz). Spawn
  the scene, print each arm's `gripper_link` position at q = 0, and confirm all three lean
  toward the centre. Also check reachability: sample random joint configs per arm, and make
  sure the EE gets within ~3 cm of the table top over the whole goal disk. If it can't,
  reduce the radius (try 0.22) and record that.
- **Cube**: `RigidObjectCfg` + `CuboidCfg` 0.04 m, mass 0.05 kg, friction ~0.8, starting at
  z = 0.02. Reset with `reset_root_state_uniform`, x/y ±0.06 around the centre, random yaw.
- **Goal**: uniform in a disk of radius 0.10 m around the centre, resampled only on reset.
  Use a small custom `CommandTerm` with a debug-vis marker, so it shows up in the videos.
- Light as in reach. `ViewerCfg` looking down at the cell centre from about 45°, so the
  training videos show all three arms.

### MDP (starting values; adjust only with a reason you write down)

- **Actions**: three `RelativeJointPositionActionCfg` terms, one per arm, scale 0.25,
  clip ±0.25: the same rule as reach, so it stays consistent with the sim2real work. The
  actuator is the matched one (0.92 rad/s, 2.9 Nm) already in `SO101_CFG`.
- **Observations**: concatenated. For each arm: joint_pos_rel, joint_vel_rel, EE position in
  the env frame. Then cube position and quaternion (env frame), cube linear velocity, goal
  xy, cube→goal vector, and the last action (18). Enable obs normalization in the PPO cfg.
- **Rewards**:
  | term | definition | weight |
  |---|---|---|
  | approach | `1 - tanh(min_arm ‖ee - cube‖ / 0.1)` | 1.0 |
  | push_coarse | `1 - tanh(‖cube_xy - goal_xy‖ / 0.3)` | 2.0 |
  | push_fine | `1 - tanh(‖cube_xy - goal_xy‖ / 0.05)` | 4.0 |
  | success | `‖cube_xy - goal_xy‖ < 0.03` | 5.0 |
  | arm_proximity | sum over arm pairs of `relu(0.08 - ‖ee_i - ee_j‖)` | -10.0 |
  | cube_dropped | cube z < -0.05 (paired with the termination) | -5.0 |
  | action_rate | `action_rate_l2` over all 18 | -0.01 |
  | joint_vel | `joint_vel_l2` summed over the 3 arms | -0.001 |
- **Terminations**: time_out (episode 10 s); cube z < -0.05 (pushed off the table).
- **Events**: reset each arm's joints by offset (-0.3, 0.3) rad (narrower than reach, so the arms
  don't spawn tangled together); stiffness randomization ×U(0.7, 1.3) per arm, as in reach.
- **Sim**: dt 0.01, decimation 2 (50 Hz policy), same as reach.

### PPO

Copy `VanillaPPORunnerCfg`, then use actor and critic `[256, 128, 64]`, obs normalization on,
`num_steps_per_env 24`, `save_interval 500`, `entropy_coef 0.005`. Leave the other algorithm
settings alone.

### Bring-up order (don't skip steps)

1. Scene only: spawn 4 envs, step zero actions for 2 s, check that nothing explodes or
   interpenetrates, and print the EE and cube positions. Commit.
2. Smoke train: `--num_envs 64 --max_iterations 10`. Must run without errors, with no
   NaNs in any reward term. Commit.
3. Throughput: `--num_envs 8192 --max_iterations 30`, and note the iteration time.
   **If the reach retrain is running at the same time, measure while it runs.**
4. **Launch** in tmux session `push`, with `--num_envs 8192 --run_name overnight --video
   --video_length 500 --video_interval 24000` (one video about every 1000 iterations) and
   `--max_iterations` set so the run **ends about 14 h after launch**, based on the
   measured iteration time.
5. Watch the first ~300 iterations. Reward should trend up, the episode length should not
   collapse (if it does, the cube is getting shoved off the table), and nothing NaN.
   If it's clearly broken, fix it and relaunch. Record every relaunch and the reason.

### Fallback

If the cell can't be made to train within ~3 hours of trying, stop working on it, write down
exactly what blocks it, and use the GPU for 2 extra seeds of the reach retrain instead
(`--seed 1`, `--seed 2`, run names `actuator_match_s1/_s2`). That still gives the user
robustness curves.

## When you're done

The final `docs/overnight_status.md` should have, at the top: a 5-line summary, the wandb
links, the run folders, the sim-check verdicts, the exported ONNX path, and what the user
should look at first.
