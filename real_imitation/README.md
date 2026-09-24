# Real-arm imitation demo (leader → follower)

Teach the SO-101 a task by demonstration: you move the **leader** arm by hand, the
**follower** copies it, and LeRobot records the follower's joints + wrist-camera video.
A policy (ACT) trained on those recordings then does the task on its own.

This is separate from `imitation/`, which is the *sim* behavioural-cloning scaffold.
Everything here runs on the Mac with both arms plugged in. Shared settings (ports, arm ids)
are in `config.py`.

## Python environment

LeRobot needs its own conda env on **Python 3.12**. The base env is Python 3.14, where
LeRobot's config parser (`draccus` 0.11.6) crashes with `TypeError: str | None is not callable`.

```bash
# one-time setup (conda-forge only, so no Anaconda ToS prompt)
conda create -n lerobot --override-channels -c conda-forge python=3.12 ffmpeg
conda activate lerobot
pip install -e ~/lerobot[feetech]

# every session
conda activate lerobot
```

Everything in this folder runs in that env. (`sim2real/` still runs in base.)

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
2. **Calibrate both arms with LeRobot** ✅ (2026-09-23). See "Calibration" below. Use
   `02_live_pose_check.py` first to line the follower up with the sim zero pose.
   Result: the follower's homing offsets moved at most 2.5° versus the pre-calibration backup,
   so the RL zero pose is intact. The servos now also enforce the swept min/max limits.
3. **Teleoperate** ✅ (2026-09-23): 157 s at 59.5 Hz with no problems. See "Teleop" below.
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
conda activate lerobot
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

## Teleop

Put both arms in roughly the same pose first: on start, the follower jumps to the leader's
pose. `max_relative_target=10` caps each step's change at 10 degrees (the calibration uses
degrees), so a pose mismatch can't throw the arm. Ctrl+C makes the follower go limp, so
support it.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5AAF2879831 \
    --robot.id=so101_follower --robot.calibration_dir=calibration/follower \
    --robot.max_relative_target=10 \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1192341 \
    --teleop.id=so101_leader --teleop.calibration_dir=calibration/leader \
    --display_data=true
```

`--display_data=true` opens LeRobot's viewer with the joint plots and the wrist-camera feed.

## Camera

The wrist camera is **OpenCV index 0** (it reports "USB2.0_CAM1" in System Information). We
record at 640x480 @ 30 fps: the policy downsamples images anyway, and 1080p over USB 2.0 slows
the control loop.

- macOS blocks the camera until your terminal app is allowed under System Settings > Privacy &
  Security > Camera. Quit and reopen the terminal after allowing it.
- **Indices can reorder** when cameras are added or removed (e.g. an iPhone Continuity Camera
  showing up). If the view looks wrong, re-run `lerobot-find-cameras opencv` and check
  `outputs/captured_images/` (gitignored).
