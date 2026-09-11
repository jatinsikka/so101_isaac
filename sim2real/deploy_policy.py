"""Run the sim-trained reach policy on the real SO-101.

Runs on the Mac (USB side). Needs: onnxruntime, numpy, scservo_sdk. No Isaac Sim.

DEFAULT IS A DRY RUN -- the arm does not move. The loop reads real sensors, runs the
real policy, and logs the joint targets it *would* send. That is the cheapest way to
catch unit, sign, ordering and zero-offset mistakes, which are the usual sim2real
failures. Only pass --allow-motion once the dry-run numbers look right.

    python deploy_policy.py --policy policy.onnx                      # dry run
    python deploy_policy.py --policy policy.onnx --allow-motion       # arm moves

POLICY I/O CONTRACT (from the trained checkpoint; ONNX reports obs[1,25] -> actions[1,6]):
    obs[0:6]   joint_pos_rel   = joint_pos - default(0)   rad
    obs[6:12]  joint_vel_rel   = joint_vel - default(0)   rad/s
    obs[12:19] pose_command    = target xyz + quat(w,x,y,z), ROBOT BASE frame
    obs[19:25] last_action     = previous raw action, clipped to [-1, 1]
    action     -> target = current_pos + ACTION_SCALE * clip(action, -1, 1)
    control rate 50 Hz (sim decimation 2 x dt 0.01)
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import onnxruntime as ort

from real_robot_interface import JOINT_NAMES, SO101Bus

CONTROL_HZ = 50.0
ACTION_SCALE = 0.25  # matches the trained ActionsCfg

# The trained action scale of 0.25 rad/step at 50 Hz is ~716 deg/s sustained, which is
# roughly twice what an STS3215 can actually deliver. Start well under it and raise only
# once the arm behaves. This is the single most important safety number here.
MAX_DELTA_RAD_PER_STEP = 0.02

# How far the integrated setpoint may lead the measured position before we stop
# advancing it. Without this, a blocked or slow joint lets the setpoint run away
# (integrator windup) and the arm lunges once it frees up.
MAX_LEAD_RAD = 0.12

# Command ranges the policy was trained on (CommandsCfg, robot base frame):
#   pos_x 0.20..0.34, pos_y -0.10..0.10, pos_z 0.05..0.25, roll 0, pitch -1.57, yaw 0
DEFAULT_TARGET_XYZ = (0.27, 0.0, 0.15)
DEFAULT_TARGET_RPY = (0.0, -1.57, 0.0)


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z), matching Isaac Lab's quat_from_euler_xyz used by UniformPoseCommand."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="path to policy.onnx")
    ap.add_argument("--port", default="/dev/tty.usbmodem5AAF2879831")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--allow-motion", action="store_true", help="actually drive the servos")
    ap.add_argument("--hold", action="store_true",
                    help="keep torque on at exit so the arm does not drop (watch temps)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--target", type=float, nargs=3, default=DEFAULT_TARGET_XYZ,
                    metavar=("X", "Y", "Z"), help="target position in the robot base frame")
    args = ap.parse_args()

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    obs_name = session.get_inputs()[0].name
    obs_shape = session.get_inputs()[0].shape
    if obs_shape[-1] != 25:
        raise SystemExit(f"expected a 25-dim observation, model wants {obs_shape}")

    quat = quat_from_euler_xyz(*DEFAULT_TARGET_RPY)
    pose_command = np.array([*args.target, *quat], dtype=np.float32)

    dt = 1.0 / CONTROL_HZ
    last_action = np.zeros(6, dtype=np.float32)

    mode = "LIVE - ARM WILL MOVE" if args.allow_motion else "DRY RUN - no motor writes"
    print(f"\n=== {mode} ===")
    print(f"target (base frame): xyz={tuple(args.target)}  quat(wxyz)={tuple(round(q, 4) for q in quat)}")
    print(f"action scale {ACTION_SCALE}, per-step delta clamped to {MAX_DELTA_RAD_PER_STEP} rad\n")

    with SO101Bus(args.port, args.baud, allow_motion=args.allow_motion,
                  release_on_close=not args.hold) as bus:
        prev_pos = np.array(bus.read_positions_rad(), dtype=np.float32)
        # Commanded setpoint, integrated from the policy's deltas. Seeded at the current
        # measured pose. See the comment at the write site for why this is not just
        # (measured_position + delta).
        setpoint = prev_pos.copy()
        if args.allow_motion:
            bus.set_torque(True)

        overruns = 0
        loop_times = []
        n_steps = int(args.seconds * CONTROL_HZ)
        print(f"{'t':>6}{'':2}" + "".join(f"{n[:9]:>10}" for n in JOINT_NAMES) + "   (rows: pos rad / target rad)")

        for step in range(n_steps):
            t0 = time.perf_counter()

            pos = np.array(bus.read_positions_rad(), dtype=np.float32)
            # Velocity by finite difference rather than the servo's PRESENT_SPEED register:
            # position and dt are known exactly, whereas the speed register's units are
            # model-dependent and easy to get silently wrong.
            vel = (pos - prev_pos) / dt
            prev_pos = pos

            obs = np.concatenate([pos, vel, pose_command, last_action]).astype(np.float32)[None, :]
            action = session.run(None, {obs_name: obs})[0].flatten()
            action = np.clip(action, -1.0, 1.0)

            delta = np.clip(action * ACTION_SCALE, -MAX_DELTA_RAD_PER_STEP, MAX_DELTA_RAD_PER_STEP)

            # Integrate onto the previous setpoint rather than onto the measured position.
            # Sim's RelativeJointPositionAction uses target = current_pos + delta, which
            # works there because the joint reaches its target within a control step. Real
            # servos lag and have a positional deadband, so re-basing on the measured
            # position issues a fresh small target every step, the servo never completes a
            # move, and the arm creeps instead of tracking. Integrating also matches what
            # the policy saw in training, since tight sim tracking makes
            # current_pos + delta ~= previous_target + delta.
            setpoint = setpoint + delta
            # anti-windup: never let the command lead the actual joint by more than this
            setpoint = np.clip(setpoint, pos - MAX_LEAD_RAD, pos + MAX_LEAD_RAD)
            targets = bus.write_targets_rad(setpoint.tolist())
            setpoint = np.array(targets, dtype=np.float32)  # adopt any joint-limit clamping
            last_action = action.astype(np.float32)

            # thermal guard once a second. Costs 6 extra serial reads so it may show up
            # as one loop overrun per second; that is a fair price for not cooking a servo.
            if step % int(CONTROL_HZ) == 0:
                ok, msg = bus.check_thermal()
                if not ok:
                    print(f"\n\nABORTING at t={step * dt:.1f}s - {msg}")
                    break

            if step % 10 == 0:
                lead = np.abs(np.array(targets) - pos).max()
                print(f"{step * dt:>6.2f}{'':2}" + "".join(f"{p:>10.3f}" for p in pos)
                      + f"   lead {lead:.3f}")
                print(f"{'':>8}" + "".join(f"{t:>10.3f}" for t in targets))

            elapsed = time.perf_counter() - t0
            loop_times.append(elapsed)
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                overruns += 1

        lt = np.array(loop_times)
        print(f"\nloop timing: mean {lt.mean()*1000:.1f} ms, max {lt.max()*1000:.1f} ms "
              f"(budget {dt*1000:.0f} ms), overruns {overruns}/{n_steps}")
        if overruns > n_steps // 20:
            print("  -> serial reads are too slow for 50 Hz; consider GroupSyncRead "
                  "or a lower CONTROL_HZ before enabling motion.")

        print("\nfinal state:")
        for n, p, t, ld in zip(JOINT_NAMES, bus.read_positions_rad(),
                               bus.read_temperatures(), bus.read_loads()):
            print(f"  {n:<15} {p:+7.3f} rad   {t:>3}C  load {ld:>4}")
        if args.hold:
            print("\n--hold: torque is STILL ON and drawing current. Release with:")
            print(f"  python -c \"from real_robot_interface import SO101Bus;"
                  f" SO101Bus('{args.port}', allow_motion=True).set_torque(False)\"")


if __name__ == "__main__":
    main()
