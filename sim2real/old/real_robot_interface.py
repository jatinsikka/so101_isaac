"""SO-101 hardware layer for policy deployment.

Runs on the machine with the USB connection (the Mac), NOT the Isaac Sim box.
Talks to the servos with scservo_sdk directly rather than through LeRobot, because
the raw SDK is already verified working on this arm and doesn't move between
LeRobot versions.

UNIT MAPPING -- measured, not assumed:
  A hand sweep of every joint gave tick spans that, compared against the URDF joint
  limits, implied 3988/4084/4140/4216/4274/4290 ticks per revolution -- all within
  ~5% of 4096. Observed per-joint midpoints were 1954/2013/1978/2013/2062, clustering
  on 2048, and the gripper rests at 2047. Hence:

      rad = (ticks - 2048) * 2*pi / 4096          (651.9 ticks/rad)

  The 2048-is-zero part is corroborated but has NOT been confirmed by physically
  posing the arm at the sim's zero pose. If joint_pos_rel looks biased in the dry
  run, that is the first thing to suspect -- fill in ZERO_OFFSET_TICKS per joint.
"""

from __future__ import annotations

import math

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# --- STS3215 control table ---
ADDR_ACCELERATION = 41
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_TEMP = 63

# STS3215 shuts down around 70 C. The zero pose holds the forearm horizontal, which is
# the worst-case gravitational moment on shoulder_lift, so holding it draws real current
# indefinitely. Abort well before the firmware limit.
TEMP_WARN_C = 45
TEMP_ABORT_C = 55

# These two are why an earlier version appeared to work but never moved the arm: the
# servos shipped with GoalSpeed=0, and in position mode an STS3215 will accept a
# GoalPosition write and then simply never travel to it. Diagnosed by reading the
# control table -- Mode was already 0 (position) and TorqueLimit 1000, so speed was
# the only thing missing.
#
# GOAL_SPEED doubles as a hard velocity cap. 600 ticks/s ~= 0.92 rad/s, which is just
# above the 1.0 rad/s that deploy_policy's per-step delta clamp allows, so the clamp
# stays the thing actually governing speed rather than this.
GOAL_SPEED = 600
ACCELERATION = 50

# --- measured conversion ---
TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * math.pi)  # 651.9
CENTER_TICKS = 2048

# Motor IDs 1..6 already match the sim's joint order, so no remapping is needed.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# Per-joint trim, in ticks, applied on top of CENTER_TICKS. Leave at 0 unless the
# dry run shows a joint reading non-zero when it is physically at the sim zero pose.
ZERO_OFFSET_TICKS = {name: 0 for name in JOINT_NAMES}

# Hard limits straight from the URDF (radians). The hand sweep went *outside* these
# on most joints, which means the URDF limits sit inside the true mechanical range --
# so clamping to them is safe. This is a backstop; deploy_policy also rate-limits.
JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex": (-1.690, 1.690),
    "wrist_flex": (-1.658, 1.658),
    "wrist_roll": (-2.744, 2.841),
    "gripper": (-0.175, 1.745),
}


def ticks_to_rad(ticks: int, joint: str) -> float:
    return (ticks - CENTER_TICKS - ZERO_OFFSET_TICKS[joint]) / TICKS_PER_RAD


def rad_to_ticks(rad: float, joint: str) -> int:
    return int(round(rad * TICKS_PER_RAD + CENTER_TICKS + ZERO_OFFSET_TICKS[joint]))


class SO101Bus:
    """Read joint positions and (optionally) write joint targets, in radians.

    Writes are refused unless allow_motion=True was passed explicitly. The default is
    read-only so the full control loop can be exercised with the arm inert.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        allow_motion: bool = False,
        release_on_close: bool = True,
    ):
        self.allow_motion = allow_motion
        self.release_on_close = release_on_close
        self.port = PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError(f"could not open {port}")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"could not set baudrate {baudrate}")
        self.packet = PacketHandler(0)

        missing = [i for i in MOTOR_IDS if self.packet.ping(self.port, i)[1] != COMM_SUCCESS]
        if missing:
            raise RuntimeError(f"servos not responding: {missing}")

    # ---------- read ----------
    def read_positions_rad(self) -> list[float]:
        """Current joint positions in radians, in sim joint order."""
        out = []
        for motor_id, name in zip(MOTOR_IDS, JOINT_NAMES):
            ticks, comm, _ = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
            if comm != COMM_SUCCESS:
                raise RuntimeError(f"read failed on {name} (id {motor_id})")
            out.append(ticks_to_rad(ticks, name))
        return out

    def read_positions_ticks(self) -> list[int]:
        out = []
        for motor_id, name in zip(MOTOR_IDS, JOINT_NAMES):
            ticks, comm, _ = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
            if comm != COMM_SUCCESS:
                raise RuntimeError(f"read failed on {name} (id {motor_id})")
            out.append(ticks)
        return out

    # ---------- write ----------
    def _w1(self, motor_id: int, addr: int, value: int, what: str) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, motor_id, addr, value)
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"write {what}={value} failed on id {motor_id} (comm={comm}, err={err})")

    def _w2(self, motor_id: int, addr: int, value: int, what: str) -> None:
        comm, err = self.packet.write2ByteTxRx(self.port, motor_id, addr, value)
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"write {what}={value} failed on id {motor_id} (comm={comm}, err={err})")

    def configure_motion(self, goal_speed: int = GOAL_SPEED, acceleration: int = ACCELERATION) -> None:
        """Give the servos a speed/acceleration budget. Without this they will not move."""
        if not self.allow_motion:
            return
        for motor_id in MOTOR_IDS:
            self._w2(motor_id, ADDR_GOAL_SPEED, goal_speed, "GoalSpeed")
            self._w1(motor_id, ADDR_ACCELERATION, acceleration, "Acceleration")

    def set_torque(self, on: bool) -> None:
        if not self.allow_motion:
            return
        for motor_id, name in zip(MOTOR_IDS, JOINT_NAMES):
            self._w1(motor_id, ADDR_TORQUE_ENABLE, 1 if on else 0, "TorqueEnable")
            # read back rather than trust the write -- a silently-ignored torque enable
            # is indistinguishable from success otherwise
            got, comm, _ = self.packet.read1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE)
            if comm == COMM_SUCCESS and got != (1 if on else 0):
                raise RuntimeError(f"TorqueEnable on {name} did not stick (wanted {int(on)}, read {got})")
        if on:
            self.configure_motion()

    def write_targets_rad(self, targets_rad: list[float]) -> list[float]:
        """Clamp targets to URDF limits and write them. Returns what was (or would be) sent.

        With allow_motion=False this clamps and returns without touching the bus, so the
        caller can log intended motion during a dry run.
        """
        clamped = []
        for value, name in zip(targets_rad, JOINT_NAMES):
            lo, hi = JOINT_LIMITS_RAD[name]
            clamped.append(min(max(value, lo), hi))

        if self.allow_motion:
            for motor_id, name, value in zip(MOTOR_IDS, JOINT_NAMES, clamped):
                self._w2(motor_id, ADDR_GOAL_POSITION, rad_to_ticks(value, name), f"GoalPosition[{name}]")
        return clamped

    def read_temperatures(self) -> list[int]:
        out = []
        for motor_id in MOTOR_IDS:
            val, comm, _ = self.packet.read1ByteTxRx(self.port, motor_id, ADDR_PRESENT_TEMP)
            out.append(val if comm == COMM_SUCCESS else -1)
        return out

    def read_loads(self) -> list[int]:
        out = []
        for motor_id in MOTOR_IDS:
            val, comm, _ = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_LOAD)
            out.append(val if comm == COMM_SUCCESS else -1)
        return out

    def check_thermal(self) -> tuple[bool, str]:
        """(ok, message). ok=False once any joint passes TEMP_ABORT_C."""
        temps = self.read_temperatures()
        hot = [(n, t) for n, t in zip(JOINT_NAMES, temps) if t >= TEMP_ABORT_C]
        if hot:
            return False, "OVER TEMP: " + ", ".join(f"{n}={t}C" for n, t in hot)
        warm = [(n, t) for n, t in zip(JOINT_NAMES, temps) if t >= TEMP_WARN_C]
        if warm:
            return True, "warm: " + ", ".join(f"{n}={t}C" for n, t in warm)
        return True, f"max {max(temps)}C"

    def close(self) -> None:
        """Release torque unless the caller asked to keep holding.

        Releasing drops the arm from wherever it is, which from the zero pose means the
        whole arm falls. Pass release_on_close=False to keep it held (e.g. to chain
        straight into a policy run) -- but then it is drawing current until something
        releases it, so watch the temperature.
        """
        try:
            if self.release_on_close:
                self.set_torque(False)
        finally:
            self.port.closePort()

    def __enter__(self) -> SO101Bus:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
