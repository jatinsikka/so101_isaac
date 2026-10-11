"""Step 2 helper: live readout of both arms' raw positions, to line up the calibration pose.

Why this exists
---------------
When `lerobot-calibrate` says "move to the middle of its range and press ENTER", the pose you
hold becomes the new 2047-tick centre for every joint. The RL deploy stack treats 2048 ticks
as the sim zero pose (verified on hardware), so for the follower we want the calibration pose
to BE that zero pose. Right now the servos' existing offsets already read ~2048 there, so:

    hold the follower so every joint reads close to 2048 here  ->  Ctrl+C  ->  calibrate
    (keep holding the same pose when calibration asks for "middle")

The leader has no RL dependence, but calibrating it in the same physical pose as the follower
makes teleop map pose-to-pose most naturally.

Read-only: torque is never touched, both arms stay limp. Ctrl+C to stop.
    python 02_live_pose_check.py
    python 02_live_pose_check.py --arm follower
"""

from __future__ import annotations

import argparse
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

import config

ADDR_PRESENT_POSITION = 56
CENTER = 2048
TICKS_PER_DEG = 4096 / 360  # 11.4 ticks per degree
OK_DEG = 5.0  # "close enough" band for lining up by eye


def open_arm(port_name: str):
    port = PortHandler(port_name)
    if not port.openPort() or not port.setBaudRate(config.BAUDRATE):
        return None
    return port


def read_all(port: PortHandler, pk: PacketHandler) -> list[int | None]:
    out = []
    for mid in config.MOTOR_IDS:
        pos, comm, err = pk.read2ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        out.append(pos if comm == COMM_SUCCESS and err == 0 else None)
    return out


def fmt(pos: int | None) -> str:
    if pos is None:
        return "   --   "
    deg = (pos - CENTER) / TICKS_PER_DEG
    mark = "ok" if abs(deg) <= OK_DEG else "  "
    return f"{deg:+6.1f}{mark}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["both", "follower", "leader"], default="both")
    args = ap.parse_args()

    pk = PacketHandler(0)
    arms = {"follower": config.FOLLOWER_PORT, "leader": config.LEADER_PORT}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}
    ports = {name: open_arm(p) for name, p in arms.items()}
    for name, port in ports.items():
        if port is None:
            print(f"warning: could not open {name} ({arms[name]})")

    print("degrees away from the 2048-tick centre; 'ok' = within", OK_DEG, "deg.  Ctrl+C to stop.\n")
    print(f"{'':10}" + "".join(f"{j[:10]:>10}" for j in config.JOINTS))
    try:
        while True:
            line = ""
            for name, port in ports.items():
                if port is not None:
                    line += f"{name:<10}" + "".join(f"{fmt(p):>10}" for p in read_all(port, pk)) + "   "
            print("\r" + line, end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        for port in ports.values():
            if port is not None:
                port.closePort()


if __name__ == "__main__":
    main()
