"""Shared settings for the real-arm imitation demo. Edit here, not in each script.

Serial ports are macOS USB names; they belong to each arm's *controller board*, so they
stay the same as long as each board stays wired to its arm. If a port disappears, run
`ls /dev/tty.usbmodem*` and tell the arms apart by supply voltage (see README).
"""

from pathlib import Path

# --- which USB port is which arm (identified 2026-09-23 by servo supply voltage) ---
FOLLOWER_PORT = "/dev/tty.usbmodem5AAF2879831"  # 12 V supply, the arm that moves
LEADER_PORT = "/dev/tty.usbmodem5A7C1192341"    # ~5 V, the arm you move by hand
BAUDRATE = 1_000_000

# LeRobot stores calibration under an id you choose; keep these fixed so recording,
# training and deployment all find the same calibration files.
FOLLOWER_ID = "so101_follower"
LEADER_ID = "so101_leader"

# Motor IDs 1-6, identical on both arms and identical to the sim's joint order.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# Where servo-register backups go (committed to git on purpose: tiny, and irreplaceable).
BACKUP_DIR = Path(__file__).parent / "backups"
