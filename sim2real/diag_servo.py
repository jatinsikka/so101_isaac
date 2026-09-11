"""Why won't the servo move? Read the STS3215 control table and check write results.

goto_zero.py commanded GOAL_POSITION and the joint never moved, with no error -- because
that code ignored write return codes. This script checks every plausible cause and
reports the comm status of each operation.

    python diag_servo.py --id 5          # read-only inspection
    python diag_servo.py --id 5 --try-move   # also attempt torque-on + a 0.05 rad nudge
"""

from __future__ import annotations

import argparse
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# STS3215 control table
REGS_1B = {
    "Mode (0=pos,1=wheel,2=pwm,3=step)": 33,
    "TorqueEnable": 40,
    "Acceleration": 41,
    "Lock": 55,
    "PresentVoltage(0.1V)": 62,
    "PresentTemp(C)": 63,
    "ServoStatus": 65,
    "Moving": 66,
}
REGS_2B = {
    "MinAngleLimit": 9,
    "MaxAngleLimit": 11,
    "GoalPosition": 42,
    "GoalTime": 44,
    "GoalSpeed": 46,
    "TorqueLimit": 48,
    "PresentPosition": 56,
    "PresentSpeed": 58,
    "PresentLoad": 60,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/tty.usbmodem5AAF2879831")
    ap.add_argument("--id", type=int, default=5)
    ap.add_argument("--try-move", action="store_true")
    args = ap.parse_args()

    port = PortHandler(args.port)
    if not port.openPort():
        raise SystemExit(f"cannot open {args.port}")
    port.setBaudRate(1_000_000)
    pk = PacketHandler(0)

    print(f"\n=== servo id {args.id} control table ===")
    for name, addr in REGS_1B.items():
        val, comm, err = pk.read1ByteTxRx(port, args.id, addr)
        status = "" if comm == COMM_SUCCESS else "  <READ FAILED>"
        print(f"  [{addr:>3}] {name:<36} {val}{status}")
    for name, addr in REGS_2B.items():
        val, comm, err = pk.read2ByteTxRx(port, args.id, addr)
        status = "" if comm == COMM_SUCCESS else "  <READ FAILED>"
        print(f"  [{addr:>3}] {name:<36} {val}{status}")

    if not args.try_move:
        print("\n(read-only; pass --try-move to attempt an actual nudge)")
        port.closePort()
        return

    print("\n=== attempting torque enable + small nudge ===")
    comm, err = pk.write1ByteTxRx(port, args.id, 40, 1)
    print(f"  write TorqueEnable=1 -> comm={comm} ({'OK' if comm == COMM_SUCCESS else 'FAIL'}) err={err}")
    val, c, _ = pk.read1ByteTxRx(port, args.id, 40)
    print(f"  read back TorqueEnable = {val}")

    # some units refuse to move with GoalSpeed/TorqueLimit at 0
    for addr, name, value in ((46, "GoalSpeed", 300), (48, "TorqueLimit", 1000)):
        cur, _, _ = pk.read2ByteTxRx(port, args.id, addr)
        if cur == 0:
            comm, err = pk.write2ByteTxRx(port, args.id, addr, value)
            print(f"  {name} was 0 -> wrote {value}: comm={comm} "
                  f"({'OK' if comm == COMM_SUCCESS else 'FAIL'})")

    start, _, _ = pk.read2ByteTxRx(port, args.id, 56)
    target = start - 30  # ~0.046 rad, small and safe
    print(f"  PresentPosition={start}, commanding GoalPosition={target}")
    comm, err = pk.write2ByteTxRx(port, args.id, 42, target)
    print(f"  write GoalPosition -> comm={comm} ({'OK' if comm == COMM_SUCCESS else 'FAIL'}) err={err}")
    readback, _, _ = pk.read2ByteTxRx(port, args.id, 42)
    print(f"  read back GoalPosition = {readback}  {'(stuck!)' if readback != target else '(accepted)'}")

    for i in range(10):
        time.sleep(0.2)
        pos, _, _ = pk.read2ByteTxRx(port, args.id, 56)
        mv, _, _ = pk.read1ByteTxRx(port, args.id, 66)
        load, _, _ = pk.read2ByteTxRx(port, args.id, 60)
        print(f"    t={i*0.2:.1f}s  pos={pos:>5}  moved={pos-start:+4d}  Moving={mv}  Load={load}")

    end, _, _ = pk.read2ByteTxRx(port, args.id, 56)
    print(f"\n  net movement: {end - start:+d} ticks")
    if end == start:
        print("  -> servo did NOT move. Check Mode, TorqueEnable readback, TorqueLimit, voltage.")
    pk.write1ByteTxRx(port, args.id, 40, 0)
    print("  torque released.")
    port.closePort()


if __name__ == "__main__":
    main()
