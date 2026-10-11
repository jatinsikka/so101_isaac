# Change log

Every change Claude makes to existing code gets an ID (C1, C2, ...). In the code:

```python
    # C2 OLD| the original line, commented out
    # >>> C2 NEW (2026-10-10)
    the replacement line(s)
    # <<< C2 NEW
```

**To undo a change (no git needed):**
1. Delete everything from `# >>> Cn NEW` to `# <<< Cn NEW`, including those two lines.
2. Find and replace `# Cn OLD| ` (with the trailing space) with nothing.

`grep -rn "C2 " .` lists every line a change touched. Undo newer changes first if two touch
the same lines.

| ID | Date | Files | What | Why |
|---|---|---|---|---|
| C1 | 2026-10-10 | `sim2real/deploy_policy.py` | Target quaternion sent as (x, y, z, w) instead of (w, x, y, z) | Isaac Lab 3.0 orders quaternions xyzw, so on hardware the policy saw the wrong target orientation. Verified in the env and in `isaaclab/utils/math.py`. |
| C2 | 2026-10-10, **reverted 2026-10-10** | `sim2real/deploy_policy.py`, `sim2real/real_robot_interface.py` (comment only) | Deploy applies actions like training: `target = measured + 0.25 * clip(action, -1, 1)`. Removed the 0.02 rad per-step clamp, the integrated setpoint and the 0.12 rad lead cap. | The run-14 policy reaches the target in sim within 6 mm, even capped at the real 0.92 rad/s, but oscillated on hardware with the old deploy rule (`live_run1.log`, `live_run2.log`). |

**C2 reverted (2026-10-10):** with C2 the arm barely moved (`live_run3.log`): the policy's actions at the start pose are ~0.06, i.e. 0.014 rad targets that the real servo's dead zone ignores. Back to the old deploy rule, the state of `live_run2.log` (oscillates). To re-apply C2: `git show 1ca9f4a:sim2real/deploy_policy.py`.

| ID | Date | Files | What | Why |
|---|---|---|---|---|
| C3 | 2026-10-10 | `sim2real/` (new: `arm.py`, `move_to.py`, `step_test.py`, `README.md`), `scripts/step_test_sim.py` | Rewrote `sim2real/` as short single-purpose scripts. Old files moved untouched to `sim2real/old/`; `scan_motors.py`, `read_joints.py`, `diag_servo.py` moved to `sim2real/tools/`. | Simpler, followable code, plus a step test that measures the sim-vs-real actuator gap. **Undo:** `git mv sim2real/old/* sim2real/tools/* sim2real/`, then delete the new files. |
| C4 | 2026-10-10 | `sim2real/arm.py` | `torque_on()` sets the servos' P gain to 32 (factory default) before engaging torque; `python arm.py` prints the P gains. | LeRobot writes P = 16 on every connect; at 16 the real elbow stopped ~0.05 rad short of every target and ignored steps under ~0.03 rad (step test, 2026-10-10). Not saved to EEPROM (lock flag 1), so a power cycle or LeRobot connect puts it back to 16. Undo: the two-step rule for C4. |
