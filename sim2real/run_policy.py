"""run_policy.py -- run the trained policy on the real arm at 50 Hz.   [C6, 2026-10-10]

Each step: read joints -> policy -> target = joint pos + 0.25 * action (training's rule)
-> send. Every step is saved to logs/run_<time>.csv.

    python move_to.py zero --move                            # training always started at zero
    python run_policy.py                                     # dry run: arm does not move
    python run_policy.py --move                              # training rule only
    python run_policy.py --move --push 0.02                  # + dead-zone push (see below)

--push: the real joint (P = 32) ignores target offsets under ~0.02 rad and stops ~0.02 rad
short, while the sim joint follows any offset (step tests, 2026-10-10). The push adds that
much to each joint's offset, in the direction the policy is asking for.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from arm import JOINTS, Arm
from policy import ACTION_SCALE, TARGET_XYZ, Policy

HZ = 50
MAX_OFFSET = 0.25  # rad; safety cap on how far a target may be from the joint (= action of +-1)
PUSH_JOINTS = np.array([1, 1, 1, 1, 1, 0])  # no push on the gripper: it is closed and would just squeeze


def offsets_from(action, push):
    """Training's rule, plus the optional dead-zone push."""
    offset = np.clip(ACTION_SCALE * action, -MAX_OFFSET, MAX_OFFSET)
    return offset + push * np.sign(offset) * PUSH_JOINTS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.onnx")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--push", type=float, default=0.0, help="dead-zone push, rad (try 0.02)")
    ap.add_argument("--target", type=float, nargs=3, default=TARGET_XYZ, metavar=("X", "Y", "Z"))
    ap.add_argument("--move", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()

    policy = Policy(args.policy, args.target)
    arm = Arm(can_move=args.move)
    pos = arm.read_joints()
    print(f"target {tuple(args.target)} m, push {args.push} rad, {'MOVING' if args.move else 'DRY RUN'}")
    if np.abs(pos).max() > 0.1:
        print(f"note: arm is {np.abs(pos).max():.2f} rad from the zero pose training started from "
              f"(python move_to.py zero --move)")
    if args.move:
        input("Clear the area, hand on the 12 V supply, then press Enter...")
        arm.torque_on()

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"run_{time.strftime('%Y%m%d-%H%M%S')}_push{args.push}.csv"
    names = [f"{kind}_{j}" for kind in ("pos", "vel", "action", "target") for j in JOINTS]
    with open(out, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["t"] + names)
        print(f"\n{'t':>5}  " + " ".join(f"{j[:8]:>8}" for j in JOINTS) + "   max|action|")

        last_action, prev = np.zeros(6, np.float32), pos
        t0 = time.perf_counter()
        for step in range(int(args.seconds * HZ)):
            tick = time.perf_counter()
            pos = arm.read_joints()
            vel = (pos - prev) * HZ
            prev = pos
            action = policy.act(pos, vel, last_action)
            target = arm.send_targets(pos + offsets_from(action, args.push))
            last_action = action

            t = tick - t0
            log.writerow([f"{t:.3f}"] + [f"{v:.4f}" for v in (*pos, *vel, *action, *target)])
            if step % (HZ // 2) == 0:  # print twice a second
                print(f"{t:>5.1f}  " + " ".join(f"{p:>+8.3f}" for p in pos) + f"   {np.abs(action).max():.3f}")
            if step % HZ == 0 and arm.too_hot():
                print("servo too hot -- stopping")
                break
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - tick)))

    print(f"\nsaved {out}")
    if args.move:
        print("Arm is HELD. Release with: python arm.py --release")
    arm.close()
