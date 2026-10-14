"""haptic_teleop.py -- leader->follower teleop where you FEEL the follower.   [CLAUDE 2026-10-14]

Idea (position-position bilateral teleop): the leader's servos are STS3215s too, so we can
turn their torque on and give them a goal. If the leader's goal is where its joint already is,
the servo pushes with zero force (you feel nothing). If the goal is where the FOLLOWER is,
the servo pulls your hand toward the follower. In free space the follower keeps up, so the gap
is tiny; when the follower hits something and stops, the gap grows and you feel it push back.
The gripper is the best demo: squeeze an object and the leader trigger resists.

Replaces lerobot-teleoperate (both want the same serial ports: never run them together).
Runs in the base env, not the lerobot env: it needs only scservo_sdk, via ../sim2real/arm.py.

    python haptic_teleop.py --check    # 1. read-only: both arms side by side, nothing moves
    python haptic_teleop.py            # 2. follower copies leader, leader stays limp
    python haptic_teleop.py --haptic   # 3. + the leader pushes back (after you write haptic_target)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# from arm import JOINTS, P_COEFFICIENT, Arm, print_joints
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim2real"))  # [CLAUDE 2026-10-14] arm.py lives there
# from arm import JOINTS, P_COEFFICIENT, Arm  # noqa: E402
from arm import GOAL_SPEED, JOINTS, P_COEFFICIENT, Arm  # noqa: E402  [CLAUDE 2026-10-14]

# LEADER_PORT = "/dev/tty.usbmodem5A7C1192341"  # ~5 V arm you move by hand (see real_imitation/config.py)
from config import LEADER_PORT  # noqa: E402  [CLAUDE 2026-10-14] one place for the port

HZ = 50
MAX_STEP_RAD = 0.10  # follower: biggest per-step change in its goal, so a leader jerk can't throw it

# >>> [CLAUDE 2026-10-14] follower feel for teleop only (arm.py keeps P=32, speed 600 for policy runs;
# arm.torque_on() rewrites those, so running a policy afterwards is unaffected)
D_COEFFICIENT = 22         # STS3215 register: D gain of the servo's position loop (damping)
FOLLOWER_P_GAIN = 16       # LeRobot's teleop value; 32 overshoots and oscillates on fast hand motion
FOLLOWER_D_GAIN = 32       # LeRobot's teleop value; damps the overshoot
FOLLOWER_SPEED = 2000      # ticks/s = 3.1 rad/s (arm.py's 600 = 0.92 rad/s lags a moving hand)
# <<< [CLAUDE 2026-10-14]

# Leader feel. Start weak; raise only once it behaves.
TORQUE_LIMIT = 48            # STS3215 RAM register: cap on output torque, 0..1000 (0.1 % units)
LEADER_TORQUE_LIMIT = 150    # 15 % of max: you can always overpower it
LEADER_P_GAIN = 8            # soft spring (arm.py uses 32 for the follower)
# DEADBAND_RAD = 0.05          # gaps smaller than this are just follower lag, not contact
DEADBAND_RAD = 0.08          # [CLAUDE 2026-10-14] resting gravity sag measured up to ~0.075
STOPPED_RAD = 0.005          # [CLAUDE 2026-10-14] follower moved less than this in one step = stuck


# >>> [CLAUDE 2026-10-14] contact = follower is stuck AND far from the leader.
# Lagging joints are still moving, so they don't count; that's what stops drag on fast moves.
def contact_mask(q_leader, q_follower, q_follower_prev):
    stopped = np.abs(q_follower - q_follower_prev) < STOPPED_RAD
    return stopped & (np.abs(q_follower - q_leader) > DEADBAND_RAD)
# <<< [CLAUDE 2026-10-14]


# def haptic_target(q_leader, q_follower):
def haptic_target(q_leader, q_follower, contact):  # [CLAUDE 2026-10-14] contact from contact_mask()
    """Where the leader servos should aim, rad, shape (6,).

    Returning q_leader          -> zero force (no haptics).
    Returning q_follower        -> full pull toward the follower (also drags on normal lag).
    TODO(you): something in between, per joint:
      - inside DEADBAND_RAD, return the leader's own position (no force);
      - outside it, aim toward the follower, so the push grows with the gap beyond the deadband.
      Tips: np.where / np.sign / np.abs do it without a loop. Plot or print the gap while
      teleoperating in free space first: the deadband must be bigger than normal lag.
      Stretch: use follower.loads() (signed, -1000..1000) instead of the position gap.
    """
    # q_l, q_f = leader.read_joints(), follower.read_joints()
    #
    # if np.abs(q_l - q_f) < DEADBAND_RAD:
    #     return q_l
    # else:
    #     return q_leader + np.sign(q_l - q_f) * (np.abs(q_l - q_f) - DEADBAND_RAD)
    # return q_leader  # placeholder: no force

    # >>> [CLAUDE 2026-10-14] your logic, made per-joint and pointed toward the follower
    gap = q_follower - q_leader                                # leader -> follower, per joint
    push = np.sign(gap) * (np.abs(gap) - DEADBAND_RAD)         # grows from 0 at the deadband edge
    # return np.where(np.abs(gap) < DEADBAND_RAD, q_leader, q_leader + push)
    return np.where(contact, q_leader + push, q_leader)  # [CLAUDE 2026-10-14]
    # <<< [CLAUDE 2026-10-14]


def check(leader, follower):
    """Pose both arms the same by eye; the 'diff' column should be near 0 on every joint."""
    while True:
        q_l, q_f = leader.read_joints(), follower.read_joints()
        print(f"\n{'joint':<14} {'leader':>8} {'follower':>9} {'diff':>7}")
        for name, a, b in zip(JOINTS, q_l, q_f):
            print(f"{name:<14} {a:+8.3f} {b:+9.3f} {a - b:+7.3f}")
        time.sleep(0.5)


def run(leader, follower, haptic):
    follower.torque_on()
    # >>> [CLAUDE 2026-10-14] override arm.py's policy-run settings with the teleop ones above
    for i in range(1, 7):
        follower._write(i, P_COEFFICIENT, 1, FOLLOWER_P_GAIN)
        follower._write(i, D_COEFFICIENT, 1, FOLLOWER_D_GAIN)
        follower._write(i, GOAL_SPEED, 2, FOLLOWER_SPEED)
    # <<< [CLAUDE 2026-10-14]
    if haptic:
        for i in range(1, 7):
            # leader._write(i, TORQUE_LIMIT, 2, LEADER_TORQUE_LIMIT)
            leader._write(i, TORQUE_LIMIT, 2, 0)  # [CLAUDE 2026-10-14] no force until a joint is in contact
        leader.torque_on()                       # holds where it is (sets P=32 ...)
        for i in range(1, 7):
            leader._write(i, P_COEFFICIENT, 1, LEADER_P_GAIN)  # ... so soften it right after
    print("running -- Ctrl+C to stop")

    period, step = 1.0 / HZ, 0
    # >>> [CLAUDE 2026-10-14]
    q_f_prev = follower.read_joints()
    force_on = np.zeros(6, dtype=bool)  # which leader joints currently have their torque limit raised
    # <<< [CLAUDE 2026-10-14]
    while True:
        t0 = time.perf_counter()
        q_l, q_f = leader.read_joints(), follower.read_joints()

        # follower chases the leader, rate-limited
        follower.send_targets(q_f + np.clip(q_l - q_f, -MAX_STEP_RAD, MAX_STEP_RAD))
        # if haptic:
        #     leader.send_targets(haptic_target(q_l, q_f))
        # >>> [CLAUDE 2026-10-14]
        if haptic:
            contact = contact_mask(q_l, q_f, q_f_prev)
            leader.send_targets(haptic_target(q_l, q_f, contact))  # target first, so raising force can't jerk
            # Torque limit 0 = truly free (a stale target would otherwise drag); only write on change.
            for j in np.flatnonzero(contact != force_on):
                leader._write(j + 1, TORQUE_LIMIT, 2, LEADER_TORQUE_LIMIT if contact[j] else 0)
            force_on = contact
        q_f_prev = q_f
        # <<< [CLAUDE 2026-10-14]

        step += 1
        if step % HZ == 0 and follower.too_hot():
            raise RuntimeError("follower servo too hot -- stopping")
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read-only side-by-side view")
    ap.add_argument("--haptic", action="store_true", help="turn on leader force feedback")
    args = ap.parse_args()

    follower = Arm(can_move=not args.check)
    leader = Arm(can_move=args.haptic, port=LEADER_PORT)  # leader never written unless --haptic
    try:
        check(leader, follower) if args.check else run(leader, follower, args.haptic)
    except KeyboardInterrupt:
        pass
    finally:
        if args.haptic:
            leader.torque_off()  # leader goes limp; the follower keeps holding its last goal
            for i in range(1, 7):
                leader._write(i, TORQUE_LIMIT, 2, 1000)  # [CLAUDE 2026-10-14] don't leave it at 0 (RAM, lasts till power-off)
        print("\nstopped. follower is still holding -- `python ../sim2real/arm.py --release` (support it) to drop it.")
        leader.close()
        follower.close()
