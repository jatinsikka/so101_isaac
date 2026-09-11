"""Slowly drive the SO-101 to its zero pose, one joint at a time.

This is the safest possible first motion: a deterministic ramp to a known pose, with
no policy involved. It exists for two reasons.

  1. The trained policy only ever saw episodes starting at all-joints-zero (the reset
     event scales zero defaults, so it is a no-op). The real arm must therefore start
     near zero or the policy is out of distribution.
  2. With torque off the arm droops, so it cannot be hand-held at zero. Torque holding
     it is the practical way to get there.

Moving ONE joint at a time is deliberate: if the tick<->radian mapping or a joint's
sign is wrong, the error shows up as one slow wrong-direction move you can stop,
rather than six at once.

    python goto_zero.py --joint gripper                     # dry run, prints the plan
    python goto_zero.py --joint gripper     --allow-motion   # actually moves
    python goto_zero.py --joint all         --allow-motion   # sequential, confirms each
"""

from __future__ import annotations

import argparse
import time

from real_robot_interface import JOINT_LIMITS_RAD, JOINT_NAMES, SO101Bus

RATE_HZ = 50.0
# 0.004 rad/step at 50 Hz = 0.2 rad/s = ~11 deg/s. Deliberately crawling.
STEP_RAD = 0.004
TOL_RAD = 0.01

# Least-consequence first: gripper is already near zero, wrists are low-inertia and far
# from the table, the shoulder carries the whole arm so it goes last.
SAFE_ORDER = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]


def ramp_joint(bus: SO101Bus, name: str, allow_motion: bool, timeout_s: float = 40.0) -> bool:
    """Walk a virtual setpoint toward zero and command that.

    Do NOT command (measured_position + small_step): the servo has a positional
    deadband of a few ticks, so a small offset from the live position is inside the
    deadband and gets ignored, and re-reading the position every loop means the request
    never gets any further ahead. The joint then sits still forever. Advancing an
    independent setpoint gives the servo a real trajectory to track.
    """
    idx = JOINT_NAMES.index(name)
    lo, hi = JOINT_LIMITS_RAD[name]
    dt = 1.0 / RATE_HZ
    deadline = time.time() + timeout_s

    start = bus.read_positions_rad()[idx]
    print(f"\n  {name}: {start:+.3f} rad -> 0.000 rad "
          f"({abs(start) / (STEP_RAD * RATE_HZ):.1f}s at {STEP_RAD * RATE_HZ:.2f} rad/s)")
    if not allow_motion:
        print("    DRY RUN - not moving. Re-run with --allow-motion.")
        return True

    setpoint = start
    stall_ref, stall_since = start, time.time()

    while time.time() < deadline:
        pos = bus.read_positions_rad()
        actual = pos[idx]
        if abs(actual) <= TOL_RAD:
            print(f"\n    reached {actual:+.4f} rad")
            return True

        # advance the setpoint toward zero, never overshooting past it
        remaining = -setpoint
        setpoint += max(-STEP_RAD, min(STEP_RAD, remaining))
        setpoint = min(max(setpoint, lo), hi)

        targets = list(pos)          # hold the other joints where they are
        targets[idx] = setpoint
        bus.write_targets_rad(targets)

        err = setpoint - actual
        # thermal check ~1 Hz: shoulder_lift holding the arm horizontal is the hot one
        if int(time.time() * 1) != int((time.time() - dt) * 1):
            ok, msg = bus.check_thermal()
            if not ok:
                print(f"\n    ABORTING - {msg}")
                return False
            thermal_note = msg
        else:
            thermal_note = ""
        print(f"\r    actual {actual:+.4f}  setpoint {setpoint:+.4f}  lag {err:+.4f} rad  {thermal_note}   ",
              end="", flush=True)

        # stall check: setpoint pulling away while the joint refuses to follow
        if abs(actual - stall_ref) > 0.02:
            stall_ref, stall_since = actual, time.time()
        elif time.time() - stall_since > 2.0 and abs(err) > 0.10:
            print(f"\n    STALLED: setpoint is {err:+.3f} rad ahead but the joint is not moving.")
            print("    Likely load/gravity, a mechanical limit, or torque not holding.")
            return False

        time.sleep(dt)

    print(f"\n    TIMEOUT after {timeout_s}s - did not reach zero.")
    print("    If it moved AWAY from zero, that joint's sign is inverted: stop and tell me.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/tty.usbmodem5AAF2879831")
    ap.add_argument("--joint", default="gripper", choices=JOINT_NAMES + ["all"])
    ap.add_argument("--allow-motion", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the per-joint confirmation")
    ap.add_argument("--hold", action="store_true",
                    help="keep torque enabled on exit so the arm does not drop (watch temps!)")
    args = ap.parse_args()

    todo = SAFE_ORDER if args.joint == "all" else [args.joint]

    with SO101Bus(args.port, allow_motion=args.allow_motion,
                  release_on_close=not args.hold) as bus:
        print("current pose:")
        for n, v in zip(JOINT_NAMES, bus.read_positions_rad()):
            print(f"  {n:<15} {v:+7.3f} rad")

        if args.allow_motion:
            bus.set_torque(True)

        for name in todo:
            if args.allow_motion and not args.yes:
                if input(f"\nmove '{name}' to zero? [y/N] ").strip().lower() != "y":
                    print("  skipped")
                    continue
            if not ramp_joint(bus, name, args.allow_motion):
                print("\nstopping - resolve the above before continuing.")
                break

        print("\nfinal pose:")
        temps, loads = bus.read_temperatures(), bus.read_loads()
        for n, v, t, ld in zip(JOINT_NAMES, bus.read_positions_rad(), temps, loads):
            flag = "" if abs(v) < 0.05 else "   <- not at zero"
            print(f"  {n:<15} {v:+7.3f} rad   {t:>3}C  load {ld:>4}{flag}")

        if args.hold:
            print("\n--hold: torque STAYS ON, the arm is holding this pose.")
            print("  It is drawing current the whole time -- the horizontal zero pose is the")
            print("  worst case for shoulder_lift. Do not leave it like this. To release:")
            print("    python -c \"from real_robot_interface import SO101Bus;"
                  " SO101Bus('%s', allow_motion=True).set_torque(False)\"" % args.port)
        else:
            print("\nTorque released, so the arm has dropped from this pose.")
            print("  Use --hold to keep it held (e.g. to chain into the policy run).")


if __name__ == "__main__":
    main()
