"""move_to.py -- slowly move the arm to a pose, one joint at a time.   [C3, 2026-10-10]

    python move_to.py zero                 # dry run: prints the plan, arm does not move
    python move_to.py zero --move          # move to the sim's zero pose (where training starts)
    python move_to.py reach --move         # move to the pose the policy ends at in sim
    python move_to.py --q 0 0 0.3 0 0 0 --move

The arm stays HELD at the end (torque on). Release it with: python arm.py --release
"""

import argparse
import time

import numpy as np

from arm import JOINTS, Arm, print_joints

POSES = {
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Where the run-14 policy settles in sim for target (0.27, 0, 0.15), gripper pitched
    # down. With a tape: the gripper should be ~27 cm in front of the base, ~16 cm above the table.
    "reach": [0.01, 0.02, 0.58, -0.43, 0.03, 0.0],
}

SPEED = 0.3  # rad/s, slow on purpose
HZ = 50
# Wrists first (light, far from the table), shoulder last (it carries the whole arm).
ORDER = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]


def move(arm, goal):
    command = arm.read_joints()  # what each joint is being told to hold
    for name in ORDER:
        j = JOINTS.index(name)
        start = command[j]
        steps = max(1, int(abs(goal[j] - start) / SPEED * HZ))
        print(f"  {name:<14} {start:+.3f} -> {goal[j]:+.3f} rad")
        for s in range(1, steps + 1):
            command[j] = start + (goal[j] - start) * s / steps  # walk the target in small steps
            arm.send_targets(command)
            time.sleep(1 / HZ)
    time.sleep(1.0)  # let the servos settle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", nargs="?", choices=POSES.keys(), default="zero")
    ap.add_argument("--q", type=float, nargs=6, help="six joint angles (rad) instead of a named pose")
    ap.add_argument("--move", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()
    goal = np.array(args.q if args.q else POSES[args.pose])

    arm = Arm(can_move=args.move)
    print("now:")
    print_joints(arm.read_joints())
    print("goal:")
    print_joints(goal)
    if not args.move:
        print("\nDRY RUN -- add --move to actually move.")
    else:
        input("\nClear the area, hand on the 12 V supply, then press Enter...")
        arm.torque_on()
        move(arm, goal)
        q = arm.read_joints()
        print("\nreached (error = reached - goal):")
        for name, got, want in zip(JOINTS, q, goal):
            print(f"  {name:<14} {got:+.3f} rad   error {got - want:+.3f}")
        print("\nArm is HELD. Release with: python arm.py --release")
    arm.close()
