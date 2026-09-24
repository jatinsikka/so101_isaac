# Real-arm imitation demo (leader → follower)

Teach the SO-101 a task by demonstration: you move the **leader** arm by hand, the
**follower** copies it, and LeRobot records the follower's joints + wrist-camera video.
A policy (ACT) trained on those recordings then does the task on its own.

This is separate from `imitation/`, which is the *sim* behavioural-cloning scaffold.
Everything here runs on the Mac with both arms plugged in. Shared settings (ports, arm ids)
are in `config.py`.

## Hardware (identified 2026-09-23)

| Arm | Port | Supply | Notes |
|---|---|---|---|
| Follower | `/dev/tty.usbmodem5AAF2879831` | 12 V | moves; wrist camera mounted |
| Leader | `/dev/tty.usbmodem5A7C1192341` | ~5 V | passive, you move it |

Both arms: 6 × STS3215, IDs 1–6 in the sim's joint order. To tell which port is which if they
ever change, read servo voltage: ~12 V is the follower, ~5 V the leader. If the follower's port
answers nothing, its 12 V supply is off.

## Steps

1. **Back up servo calibration**: `python 01_backup_servo_calibration.py --arm follower`
   (and `--arm leader`). LeRobot calibration rewrites EEPROM offsets; this keeps the old ones.
   Backups land in `backups/` and are committed. Undo with `--restore backups/<file>.json`.
2. **Calibrate both arms with LeRobot**. See "Calibration" below. Use `02_live_pose_check.py`
   first to line the follower up with the sim zero pose.
3. Teleoperate. *(next)*
4. Record demonstrations. *(next)*
5. Train ACT, then run the policy. *(next)*

## Calibration

Calibration is interactive (it waits for ENTER), so run it in your own terminal, one arm at a
time. The calibration files are written into `calibration/` in this folder (via
`calibration_dir`) rather than LeRobot's hidden cache, so they're versioned with the repo.

**Why the pose matters.** "Middle of range" becomes each joint's new centre tick. For the
follower, hold it at the **sim zero pose** (the pose `sim2real/goto_zero.py` drives to; every
joint reads ~0° in `02_live_pose_check.py`). That keeps the RL deploy stack's
2048-ticks-is-zero assumption true. Use the same physical pose for the leader.

```bash
cd ~/so101_isaac/real_imitation
python 02_live_pose_check.py --arm follower   # line up, Ctrl+C, keep holding the pose

lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5AAF2879831 \
    --robot.id=so101_follower --robot.calibration_dir=calibration/follower
# 1) ENTER while holding the zero pose
# 2) sweep EVERY joint slowly through its full range, then ENTER

lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1192341 \
    --teleop.id=so101_leader --teleop.calibration_dir=calibration/leader
```

During the range sweep, go gently to each mechanical stop; the recorded min/max become the
servo's position limits, so a joint you don't sweep fully will be clamped short later.
