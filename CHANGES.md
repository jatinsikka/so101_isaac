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
| C2 | 2026-10-10 | `sim2real/deploy_policy.py`, `sim2real/real_robot_interface.py` (comment only) | Deploy applies actions like training: `target = measured + 0.25 * clip(action, -1, 1)`. Removed the 0.02 rad per-step clamp, the integrated setpoint and the 0.12 rad lead cap. | The run-14 policy reaches the target in sim within 6 mm, even capped at the real 0.92 rad/s, but oscillated on hardware with the old deploy rule (`live_run1.log`, `live_run2.log`). |
