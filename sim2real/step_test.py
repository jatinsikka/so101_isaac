"""step_test.py -- how does a real joint respond to a command?   [C3, 2026-10-10]

Holds the arm still, then moves ONE joint's target by a fixed amount and records the
joint for 1 s; then puts the target back and records 1 s more. Repeats for several
step sizes. The same test runs in sim (scripts/step_test_sim.py on AWS), so the two can
be compared side by side: that comparison is the sim-vs-real actuator gap.

What to look for:
  - smallest step that makes the joint move at all  (the servo's dead zone)
  - how long it takes to get there, and whether it overshoots
  - where it stops (does it reach the target?)

    python move_to.py zero --move                  # start from the sim's zero pose
    python step_test.py                            # dry run, prints the plan
    python step_test.py --joint elbow_flex --move  # run it -> logs/step_elbow_flex_<time>.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from arm import JOINTS, Arm

STEPS = [0.01, 0.02, 0.05, 0.1, 0.2]  # rad (0.2 rad = 11.5 deg)
RECORD_S = 1.0
HZ = 50


def record(arm, command, seconds):
    """Hold `command` and record (time, positions, loads) every 1/HZ s."""
    rows, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        arm.send_targets(command)
        rows.append((time.perf_counter() - t0, arm.read_joints(), arm.loads()))
        time.sleep(1 / HZ)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="elbow_flex", choices=JOINTS)
    ap.add_argument("--move", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()
    j = JOINTS.index(args.joint)

    arm = Arm(can_move=args.move)
    hold = arm.read_joints()
    print(f"{args.joint} starts at {hold[j]:+.3f} rad; steps (rad): {STEPS}")
    if not args.move:
        print("DRY RUN -- add --move to run the test.")
        raise SystemExit

    input("Arm will twitch one joint by up to 0.2 rad. Clear the area, then press Enter...")
    arm.torque_on()
    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"step_{args.joint}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step_rad", "phase", "t", "target", "pos", "load"])
        print(f"\n{'step':>6} {'reached':>8} {'time to 90%':>12} {'back to':>8}")
        for step in STEPS:
            target = hold.copy()
            target[j] += step
            up = record(arm, target, RECORD_S)   # target moved by `step`
            back = record(arm, hold, RECORD_S)   # target put back
            for phase, rows, tgt in (("up", up, target[j]), ("back", back, hold[j])):
                for t, q, load in rows:
                    log.writerow([step, phase, f"{t:.3f}", f"{tgt:.4f}", f"{q[j]:.4f}", load[j]])
            moved = np.array([q[j] for _, q, _ in up]) - hold[j]
            hit = np.nonzero(moved >= 0.9 * step)[0]
            t90 = f"{up[hit[0]][0]:.2f} s" if len(hit) else "never"
            print(f"{step:>6.2f} {moved[-1] / step:>7.0%} {t90:>12} {back[-1][1][j] - hold[j]:>+8.3f}")
    print(f"\nsaved {out}\nArm is HELD. Release with: python arm.py --release")
    arm.close()
