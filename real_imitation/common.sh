# Shared shell settings for the real_imitation scripts. Sourced, not run.
# Mirrors config.py (the Python side); if a port changes, update both.

FOLLOWER_PORT=/dev/tty.usbmodem5AAF2879831   # 12 V arm with the wrist camera
LEADER_PORT=/dev/tty.usbmodem5A7C1192341     # ~5 V arm you move by hand

# Wrist camera: OpenCV index 0, recorded small (the policy downsamples anyway, and 1080p
# over USB 2.0 slows the control loop). Re-check the index if cameras are added or removed.
CAMERAS="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"

# Arguments every LeRobot command here needs, so each script only adds what's specific to it.
# Calibration files live in ./calibration (see README), so always run from this folder.
ROBOT_ARGS=(
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT"
    --robot.id=so101_follower --robot.calibration_dir=calibration/follower
    --robot.cameras="$CAMERAS"
)
TELEOP_ARGS=(
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT"
    --teleop.id=so101_leader --teleop.calibration_dir=calibration/leader
)

# Fail early with a clear message instead of a LeRobot traceback.
cd "$(dirname "$0")" || exit 1
if [[ "$CONDA_DEFAULT_ENV" != "lerobot" ]]; then
    echo "Run 'conda activate lerobot' first (LeRobot breaks on the base env's Python 3.14)." >&2
    exit 1
fi
for p in "$FOLLOWER_PORT" "$LEADER_PORT"; do
    [[ -e "$p" ]] || { echo "Missing $p -- is that arm plugged in?" >&2; exit 1; }
done
