"""arm.py -- the only file that talks to the SO-101 servos.   [C3, 2026-10-10]

Everything else imports this. Joint values in and out are always RADIANS, in the sim's
joint order (motor IDs 1-6 already match it).

    python arm.py            # print joint positions (read-only, arm never moves)
    python arm.py --release  # turn torque off (SUPPORT THE ARM: it drops)
"""

import argparse
import math

import numpy as np
from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# ---------------- hardware facts (all measured on this arm) ----------------
PORT = "/dev/tty.usbmodem5AAF2879831"  # follower arm (12 V)
BAUD = 1_000_000
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDS = [1, 2, 3, 4, 5, 6]

TICKS_PER_RAD = 4096 / (2 * math.pi)  # 651.9; signs verified per joint on 2026-09-23
ZERO_TICK = 2048                      # tick 2048 = the sim's 0 rad on every joint

# Joint limits from the URDF (rad). Every target is clamped to these before it is sent.
LIMITS = np.array([
    [-1.920, 1.920],   # shoulder_pan
    [-1.745, 1.745],   # shoulder_lift
    [-1.690, 1.690],   # elbow_flex
    [-1.658, 1.658],   # wrist_flex
    [-2.744, 2.841],   # wrist_roll
    [-0.175, 1.745],   # gripper
])

SPEED = 600       # ticks/s, the servo's own speed cap: 600 / 651.9 = 0.92 rad/s
ACCEL = 50        # servo acceleration setting
MAX_TEMP_C = 55   # stop before the servos' ~70 C shutdown
# >>> C4 NEW (2026-10-10): servo stiffness
# P gain of the servo's position loop: push = P x error. Factory default 32; LeRobot writes 16
# on every connect (smoother teleop), which left the elbow stopping ~0.05 rad short of its
# target (step test, 2026-10-10). Set on every torque_on, so it is right whoever touched it last.
# C5 OLD| P_GAIN = 32
# >>> C5 NEW (2026-10-10): P = 32 still left a 0.02 rad gap; try 64 (watch for buzzing)
P_GAIN = 64
# <<< C5 NEW
# <<< C4 NEW

# STS3215 register addresses
TORQUE_ENABLE, ACCELERATION, GOAL_POSITION, GOAL_SPEED = 40, 41, 42, 46
# >>> C4 NEW (2026-10-10)
P_COEFFICIENT = 21
# <<< C4 NEW
PRESENT_POSITION, PRESENT_LOAD, PRESENT_TEMP = 56, 60, 63


def to_rad(ticks):
    return (np.asarray(ticks, dtype=float) - ZERO_TICK) / TICKS_PER_RAD


def to_ticks(rad):
    return np.round(np.asarray(rad, dtype=float) * TICKS_PER_RAD + ZERO_TICK).astype(int)


class Arm:
    """Open the bus once, then read joints and send targets.

    can_move=False (the default) makes every write a no-op, so any script can run as a
    dry run without the arm moving.
    """

    def __init__(self, can_move=False):
        self.can_move = can_move
        self.port = PortHandler(PORT)
        if not (self.port.openPort() and self.port.setBaudRate(BAUD)):
            raise RuntimeError(f"cannot open {PORT} -- is the follower plugged in?")
        self.bus = PacketHandler(0)

    # ---- low level: one register on one servo, retried once (the bus drops a packet now and then)
    def _read(self, motor_id, addr, size):
        fn = self.bus.read1ByteTxRx if size == 1 else self.bus.read2ByteTxRx
        for _ in range(2):
            value, comm, _ = fn(self.port, motor_id, addr)
            if comm == COMM_SUCCESS:
                return value
        raise RuntimeError(f"no reply from servo {motor_id} (register {addr}) -- is the 12 V supply on?")

    def _write(self, motor_id, addr, size, value):
        if not self.can_move:
            return
        fn = self.bus.write1ByteTxRx if size == 1 else self.bus.write2ByteTxRx
        for _ in range(2):
            comm, _ = fn(self.port, motor_id, addr, int(value))
            if comm == COMM_SUCCESS:
                return
        raise RuntimeError(f"write to servo {motor_id} (register {addr}) failed twice")

    # ---- what the other scripts use
    def read_joints(self):
        """Joint positions, rad, shape (6,)."""
        return to_rad([self._read(i, PRESENT_POSITION, 2) for i in IDS])

    def send_targets(self, targets):
        """Clamp to the joint limits, send, and return what was sent (rad)."""
        targets = np.clip(targets, LIMITS[:, 0], LIMITS[:, 1])
        for motor_id, ticks in zip(IDS, to_ticks(targets)):
            self._write(motor_id, GOAL_POSITION, 2, ticks)
        return targets

    def torque_on(self):
        """Hold the arm where it is right now, with the speed cap set."""
        here = self.read_joints()
        for motor_id, ticks in zip(IDS, to_ticks(here)):
            self._write(motor_id, GOAL_POSITION, 2, ticks)  # target = here, so nothing jumps
            self._write(motor_id, GOAL_SPEED, 2, SPEED)     # without a speed the servo never moves
            self._write(motor_id, ACCELERATION, 1, ACCEL)
            # >>> C4 NEW (2026-10-10)
            self._write(motor_id, P_COEFFICIENT, 1, P_GAIN)  # stiffer position loop
            # <<< C4 NEW
            self._write(motor_id, TORQUE_ENABLE, 1, 1)

    def torque_off(self):
        for motor_id in IDS:
            self._write(motor_id, TORQUE_ENABLE, 1, 0)

    def temperatures(self):
        return [self._read(i, PRESENT_TEMP, 1) for i in IDS]

    def loads(self):
        """Signed load, -1000..1000 (0.1 % of max torque). Bit 10 of the register is the sign."""
        raw = [self._read(i, PRESENT_LOAD, 2) for i in IDS]
        return [-(r & 1023) if r & 1024 else (r & 1023) for r in raw]

    def too_hot(self):
        return max(self.temperatures()) >= MAX_TEMP_C

    def close(self):
        self.port.closePort()


def print_joints(q):
    for name, value in zip(JOINTS, q):
        print(f"  {name:<14} {value:+.3f} rad  ({math.degrees(value):+6.1f} deg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="turn torque off (the arm drops!)")
    args = ap.parse_args()

    arm = Arm(can_move=args.release)
    if args.release:
        input("Torque OFF: support the arm, then press Enter...")
        arm.torque_off()
        print("released")
    else:
        print_joints(arm.read_joints())
        print("  temps (C):", arm.temperatures(), "  loads:", arm.loads())
        # >>> C4 NEW (2026-10-10)
        print("  P gains:  ", [arm._read(i, P_COEFFICIENT, 1) for i in IDS])
        # <<< C4 NEW
    arm.close()
