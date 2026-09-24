"""Step 1: save the follower's calibration registers BEFORE LeRobot calibration rewrites them.

Why this exists
---------------
`lerobot-calibrate` writes three registers into each servo's EEPROM (memory that survives
power-off):

    Homing_Offset       Present_Position = raw_encoder - Homing_Offset. LeRobot sets it so
                        whatever pose you hold at "middle of range" reads 2047.
    Min/Max_Position_Limit   the servo refuses goal positions outside these.

The RL deploy stack (sim2real/real_robot_interface.py) assumes 2048 ticks == the sim zero pose,
and we verified that on hardware today. If calibration is done in a different pose, that
mapping silently shifts. With this backup the old values can always be put back.

Usage (read-only unless --restore):
    python 01_backup_servo_calibration.py                    # back up the follower
    python 01_backup_servo_calibration.py --arm leader       # back up the leader
    python 01_backup_servo_calibration.py --restore backups/follower_<time>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

import config

# STS3215 control table: name -> (address, size in bytes). Taken from LeRobot's
# lerobot/motors/feetech/tables.py (STS_SMS_SERIES_CONTROL_TABLE), not from memory.
CALIBRATION_REGS = {
    "Homing_Offset": (31, 2),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
}
# Read for context only, never restored: useful to see what state the arm was in.
CONTEXT_REGS = {
    "Present_Position": (56, 2),
    "Operating_Mode": (33, 1),
    "P_Coefficient": (21, 1),
}
ADDR_TORQUE_ENABLE = 40
ADDR_LOCK = 55  # EEPROM write-protect: 0 = writable, 1 = locked


def open_bus(port_name: str) -> tuple[PortHandler, PacketHandler]:
    port = PortHandler(port_name)
    if not port.openPort() or not port.setBaudRate(config.BAUDRATE):
        sys.exit(f"could not open {port_name} -- is the arm plugged in (and powered, for the follower)?")
    return port, PacketHandler(0)  # protocol 0 = STS/SMS series


def read_reg(pk: PacketHandler, port: PortHandler, motor_id: int, addr: int, size: int) -> int:
    read = pk.read2ByteTxRx if size == 2 else pk.read1ByteTxRx
    value, comm, err = read(port, motor_id, addr)
    if comm != COMM_SUCCESS or err != 0:
        sys.exit(f"read failed: motor {motor_id} addr {addr} ({pk.getTxRxResult(comm)}, err {err})")
    return value


def write_reg(pk: PacketHandler, port: PortHandler, motor_id: int, addr: int, size: int, value: int) -> None:
    write = pk.write2ByteTxRx if size == 2 else pk.write1ByteTxRx
    comm, err = write(port, motor_id, addr, value)
    if comm != COMM_SUCCESS or err != 0:
        sys.exit(f"write failed: motor {motor_id} addr {addr} ({pk.getTxRxResult(comm)}, err {err})")


def backup(arm: str) -> None:
    port_name = config.FOLLOWER_PORT if arm == "follower" else config.LEADER_PORT
    port, pk = open_bus(port_name)
    # Raw register values are stored as-is (Homing_Offset is sign-magnitude encoded on the
    # wire); restoring writes the same raw value back, so no decoding is needed.
    snapshot = {"arm": arm, "port": port_name, "taken": datetime.now().isoformat(timespec="seconds"),
                "motors": {}}
    for name, mid in zip(config.JOINTS, config.MOTOR_IDS):
        regs = {r: read_reg(pk, port, mid, a, s) for r, (a, s) in {**CALIBRATION_REGS, **CONTEXT_REGS}.items()}
        snapshot["motors"][name] = {"id": mid, **regs}
        print(f"  {name:<14} " + "  ".join(f"{k}={v}" for k, v in regs.items()))
    port.closePort()

    config.BACKUP_DIR.mkdir(exist_ok=True)
    out = config.BACKUP_DIR / f"{arm}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"\nsaved {out}")


def restore(path: str) -> None:
    snapshot = json.loads(open(path).read())
    port, pk = open_bus(snapshot["port"])
    print(f"restoring {snapshot['arm']} registers from {snapshot['taken']}")
    for name, regs in snapshot["motors"].items():
        mid = regs["id"]
        # EEPROM writes need torque off and the lock cleared; re-lock afterwards.
        write_reg(pk, port, mid, ADDR_TORQUE_ENABLE, 1, 0)
        write_reg(pk, port, mid, ADDR_LOCK, 1, 0)
        for reg, (addr, size) in CALIBRATION_REGS.items():
            write_reg(pk, port, mid, addr, size, regs[reg])
        write_reg(pk, port, mid, ADDR_LOCK, 1, 1)
        print(f"  {name:<14} restored " + "  ".join(f"{r}={regs[r]}" for r in CALIBRATION_REGS))
    port.closePort()
    print("done. LeRobot's calibration JSON for this arm no longer matches -- delete or redo it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["follower", "leader"], default="follower")
    ap.add_argument("--restore", metavar="BACKUP_JSON", help="write a backup's registers back to the servos")
    args = ap.parse_args()
    restore(args.restore) if args.restore else backup(args.arm)
